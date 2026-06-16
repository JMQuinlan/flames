#!/bin/bash

#SBATCH --job-name=ShockDroplet
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 20:00:00

module purge
module load gnu9 mpich

#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/RockULikeAHuricane
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/1mm_Droplet
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/1mm_Droplet_5Ma_H20
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/1mm_Droplet_5Ma_n-Pentane
srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/input_R22_Braconnier
