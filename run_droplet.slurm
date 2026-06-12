#!/bin/bash
#----------------------------------------------------
# Example Slurm job script
# for UCCS Incline nodes
#
#   *** Hybrid Job on Normal Queue ***
#
#       This sample script specifies:
#         10 nodes (capital N)
#         160 total MPI tasks (lower case n); this is 16 tasks/node
#         8 OpenMP threads per MPI task (128 threads per node)
#
# Last revised: 20220215
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_VKVS.slurm" on Incline login node.
#
#   -- In most cases it's best to keep
#      ( MPI ranks per node ) x ( threads per rank )
#      to a number no more than 128 (total cores).
#
#   -- If you're running out of memory, try running
#      fewer tasks and/or threads per node to give each
#      process access to more memory.
#
#   -- IMPI and MVAPICH2 both do sensible process pinning by default.
#
#----------------------------------------------------
#
# JMQ 20210929 - Setting up for a detonation test case
#
#
#----------------------------------------------------



#SBATCH -J droplet        # Job name
#SBATCH -o ./logs/jmq_droplet.o%j    # Name of stdout output file
#SBATCH -e ./logs/jmq_droplet.e%j    # Name of stderr error file
#SBATCH -p compute-long         # Queue (partition) name
#SBATCH --ntasks-per-node=128
#SBATCH -N 4               # Total # of nodes
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)

# Other commands must follow all #SBATCH directives...

# module list
pwd
date
cat run_droplet.slurm
# cat ./tests/FlowVortexShedLarge/input
# cat ./tests/FlowShockDroplet/Dodecane_Case_M6.5_Vap
cat ./tests/FlowShockDroplet/Air-Dodecane-Ma6_5

# Set thread count (default value is 1)...

export OMP_NUM_THREADS=1

# Launch MPI code...

# mpirun ./bin/hydro2-2d-g++ ./tests/FlowShockDroplet/Dodecane_Case_M6.5_Vap
mpirun ./bin/hydro2-2d-g++ ./tests/FlowShockDroplet/Air-Dodecane-Ma6_5

# ---------------------------------------------------
