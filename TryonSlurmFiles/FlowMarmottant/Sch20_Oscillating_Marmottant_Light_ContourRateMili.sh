#!/bin/bash

#SBATCH --job-name=MarmLightCRMili
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

## 3D Variants ##
# Test deck ContourRate: see the header of the input file for what it changes and
# the Keller-Miksis numbers it should be compared against.
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/input_Sch20-Oscillating_Marmottant_Light_ContourRateMili
