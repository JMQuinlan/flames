#!/bin/bash
#----------------------------------------------------
# Slurm job script
# for UCCS Incline nodes
#
#   *** Job on Normal Queue ***
#
#       This sample script specifies:
#         4 nodes (capital N)
#         128 tasks/node (--ntasks-per-node)
#         1 OpenMP threads per MPI task (so a total
#               of 128 threads per node)
#
# Last revised: 20250513
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_XXX.slurm" on Incline login node.
#
#   -- In most cases it's best to keep
#      ( MPI ranks per node ) x ( threads per rank )
#      to a number no more than 128 (total cores).
#
#   -- If you're running out of memory, try running
#      fewer tasks and/or threads per node to give each
#      process access to more memory.
#
#----------------------------------------------------

#SBATCH -J Droplet              # Job name
#SBATCH -o logRuns/Droplet.o%j          # Name of stdout output file
#SBATCH -e logRuns/Droplet.e%j          # Name of stderr error file
## #SBATCH -p compute-long             # Queue (partition) name
## #SBATCH -p bigmem             # Queue (partition) name
#SBATCH -p compute             # Queue (partition) name
#SBATCH --ntasks-per-node=128  # MPI tasks per node
#SBATCH -N 4                   # Total # of nodes
#SBATCH -t 24:00:00            # Run time (hh:mm:ss)

# Other commands must follow all #SBATCH directives...

# module list
pwd
date
cat run_Droplet.slurm
# cat ./tests/FlowTopoChange/input_short
cat ./tests/FlowTopoChange/input_const_p_rho_gamma
git --git-dir .git show -s

# insure output folder is read to be writen to
mkdir -p /mmfs1/home/jquinlan/runs/RDE/droplet/droplet_20250820/short_output

# Set thread count (default value is 1)...

export OMP_NUM_THREADS=1

# Launch MPI code...

# mpirun ./bin/hydro2-2d-g++ ./tests/FlowTopoChange/input_short
mpirun ./bin/hydro2-2d-g++ ./tests/FlowTopoChange/input_const_p_rho_gamma

# ---------------------------------------------------

