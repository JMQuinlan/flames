#!/bin/bash

#SBATCH --job-name=RPE-KM
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 10:00:00

module purge
module load gnu9 mpich

#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_Neumann
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_Neumann_Large
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_NSCBC
srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_NSCBC_Large
