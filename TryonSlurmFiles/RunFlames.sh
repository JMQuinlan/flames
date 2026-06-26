#!/bin/bash

#SBATCH --job-name=RPE-KM
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

## 2D Varients ##
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_Neumann
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_NSCBC
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_Neumann_Large
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_NSCBC_Large

#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Collapsing_Neumann
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Collapsing_NSCBC
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Collapsing_Neumann_Large
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Collapsing_NSCBC_Large


## 3D Varients ##
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_Neumann_3D
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_NSCBC_3D
srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_Neumann_Large_3D
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_NSCBC_Large_3D

#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Collapsing_Neumann_3D
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Collapsing_NSCBC_3D
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Collapsing_Neumann_Large_3D
#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Collapsing_NSCBC_Large_3D
