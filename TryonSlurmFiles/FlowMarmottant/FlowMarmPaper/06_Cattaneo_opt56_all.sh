#!/bin/bash

#SBATCH --job-name=MP06all
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 500:00:00

module purge
module load gnu9 mpich

## MarmPaper 06_Cattaneo: all four option runs on ONE node, 32 tasks each ##
# HighPa/LowPa x opt5 (shell.slip = 5) / opt6 (shell.slip = 0). Same decks as the single-run
# scripts 06_Cattaneo_<HighPa,LowPa>_opt<5,6>.sh; each writes its own output_..._opt<5,6>.
# Each run only has a few dozen AMR boxes per level, so 32 tasks each is about as fast as 128.
# --exact gives each srun step its own 32 cores (Slurm >= 21.08; on older Slurm use --exclusive).
cd /home/ttryon/flames

D=/home/ttryon/flames/tests/FlowMarmottant/MarmPaper
for deck in input_06_Cattaneo_HighPa_opt5 input_06_Cattaneo_HighPa_opt6 \
            input_06_Cattaneo_LowPa_opt5  input_06_Cattaneo_LowPa_opt6; do
    srun --mpi=pmi2 --exact -n 32 --cpu-bind=cores \
        /home/ttryon/flames/bin/hydro2-3d-g++ $D/$deck > /mmfs1/home/ttryon/FLAMES_out/${deck}_${SLURM_JOB_ID}.log 2>&1 &
done
wait
