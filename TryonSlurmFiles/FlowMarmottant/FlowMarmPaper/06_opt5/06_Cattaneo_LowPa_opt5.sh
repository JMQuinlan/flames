#!/bin/bash

#SBATCH --job-name=MP06Loo5
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 500:00:00

module purge
module load gnu9 mpich

## MarmPaper 06_Cattaneo_LowPa, OPTION 5 shell transport (level-set Gamma; see README.md in this folder) ##
# Binary: built from HEAD + option5.patch and copied to bin_opt5 (README.md).
# shell.kappa_s_alpha=0: constant kappa_s (the paper's shell model), same as the local A/B runs.
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin_opt5/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/MarmPaper/input_06_Cattaneo_LowPa \
    plot_file=/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper/output_06_Cattaneo_LowPa_opt5 \
    shell.kappa_s_alpha=0.0
