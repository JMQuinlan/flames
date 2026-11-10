#!/bin/bash

#SBATCH --job-name=ImpactA30
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

## FlowTryonFun/Impact_AOA30: Ma 10, 15 deg half-angle wedge entering water on a path 30 deg from vertical (AOA 30; 2D, wedge frame, full width; see deck header) ##
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowTryonFun/Impact/impact_Impact_AOA30
