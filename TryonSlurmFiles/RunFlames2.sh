#!/bin/bash

#SBATCH --job-name=ShockDroplet
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

# Shock Droplet Interaction #
#srun  --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/RockULikeAHuricane
#srun  --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/1mm_Droplet_amr
#srun  --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/1mm_Droplet_5Ma_H20
#srun  --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/1mm_Droplet_5Ma_n-Pentane
srun  --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/1mm_Droplet_moreAMR
