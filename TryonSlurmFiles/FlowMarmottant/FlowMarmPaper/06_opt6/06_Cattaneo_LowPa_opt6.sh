#!/bin/bash

#SBATCH --job-name=MP06Loo6
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 500:00:00

module purge
module load gnu9 mpich

## MarmPaper 06_Cattaneo_LowPa, OPTION 6 with SLIP OFF = Gamma row with raw div_s u only (see ../README_06_options.md) ##
# Binary: built from HEAD + option6.patch and copied to bin_opt6noslip (README.md).
# shell.kappa_s_alpha=0: constant kappa_s (the paper's shell model), same as the local A/B runs.
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin_opt6noslip/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/MarmPaper/input_06_Cattaneo_LowPa \
    plot_file=/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper/output_06_Cattaneo_LowPa_opt6noslip \
    shell.kappa_s_alpha=0.0
