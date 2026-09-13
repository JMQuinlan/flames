#!/bin/bash
# 3D Marmottant static-Laplace bubble (octant), R0 = 1.150 mm, ruptured branch.
# Input : tests/FlowMarmottant/3D_Bubble/input_R001150000   (stop_time = 20 ms)
# Output: /mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/3D_Bubble/output_R001150000
#
# Cost (measured locally, 4 ranks, contended): ~0.68 coarse steps/s; 20 ms is
# ~1.45M steps -> ~590 h on 4 ranks.  Extrapolated: ~35-45 h on 128 ranks,
# ~100+ h on 32 ranks (would hit the walltime).  If it does, resubmit with
#   restart=/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/3D_Bubble/output_R001150000/NNNNNcell
# appended to the srun line (loses <= one plot_dt = 1 ms).
#
#SBATCH --job-name=M3D_1150000
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

EXE=/home/ttryon/flames/bin/hydro2-3d-g++
INP=/home/ttryon/flames/tests/FlowMarmottant/3D_Bubble/input_R001150000
mkdir -p /mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/3D_Bubble

# The frequency analysis needs thermo.dat, which only exists in binaries built
# after the Hydro2 gas-volume time series was added.  Warn loudly, don't abort.
if ! grep -aq "thermo.dat: gas_volume" "$EXE"; then
  echo "WARNING: $EXE predates the Hydro2 thermo.dat time series -- rebuild (--dim=3)."
  echo "         Run continues; FFT will fall back to (aliased) plotfile sampling."
fi

srun --mpi=pmi2 "$EXE" "$INP"
