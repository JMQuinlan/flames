#!/bin/bash

#SBATCH --job-name=2mmMa5.0
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/2mm_WATER_2D/2mm_Droplet_5-0Ma
