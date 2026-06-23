#!/bin/bash
#
# =============================================================================
# ANGLED-3D GARRICK RIEMANN SUITE  --  single sequential SLURM job (INCLINE)
# =============================================================================
# Runs the 20 oriented Garrick Riemann problems (4 planes x 5 angles) one after
# the next on the /mmfs1 high-speed drive.  Multi-dimension coupling check: the
# 1D profile projected onto each propagation direction n must be identical for
# every orientation.  A failing case is logged and SKIPPED so one bad run never
# kills the sweep.
#
# After the job finishes, overlay + error-table with:
#     cd tests/FlowRiemannUnitTests/Angled_3D/reference
#     python RiemannGarrickCompare_3D.py
# (its output path defaults to /mmfs1, auto-falling-back to a desktop copy.)
#
# Submit:   sbatch TryonSlurmFiles/AngledRiemann3D/RunAngled3D.sh
# =============================================================================

#SBATCH --job-name=Angled3D
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/angled3d_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/angled3d_%j_stderr
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 23:00:00

module purge
module load gnu9 mpich

NRANKS=128

EXE=/home/ttryon/flames/bin/hydro2-3d-g++
IN=/home/ttryon/flames/tests/FlowRiemannUnitTests/Angled_3D

PLANES=(XY YZ XZ XYZ)
ANGLES=(00 30 45 60 90)

# Run one case; never abort the suite on a single failure.
RUN() {
    echo
    echo "============================================================"
    echo "=== $(date +%T)   $(basename "$1")"
    echo "============================================================"
    local t0=$SECONDS
    srun -n "$NRANKS" --mpi=pmi2 "$EXE" "$1" || echo "  [FAILED] $1"
    echo "--- elapsed: $((SECONDS - t0)) s"
}

suite_start=$SECONDS

for plane in "${PLANES[@]}"; do
    for angle in "${ANGLES[@]}"; do
        RUN "$IN/input_Garrick_${plane}_${angle}"
    done
done

echo
echo "=== ANGLED-3D SUITE COMPLETE   (total: $((SECONDS - suite_start)) s) ==="
echo "    analyze: cd tests/FlowRiemannUnitTests/Angled_3D/reference && python RiemannGarrickCompare_3D.py"
