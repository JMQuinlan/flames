#!/bin/bash
# Axis-jet discriminators for the Lim (THINC + MUSCL2 + eta_consistent_advect) collapse (2026-10-07).
# All start from the Lim deck and stop at 3.2e-4 s (~1.19 t_c): the jets are fully formed by 1.05 t_c.
#   CASE=halfz   z full (-2..2), x and y still mirrored.  The +-z axes are still the corner of two
#                reflect planes (x=0, y=0); the +-x and +-y axes now touch only ONE reflect plane and
#                run through the interior z=0 grid line.  Jets on all six axes -> grid alignment;
#                jets only on +-z -> the two-plane symmetry corner.  2x the octant cost.
#   CASE=nothinc MUSCL2 + consistent eta, THINC off         -> is THINC the amplifier?
#   CASE=thincdonor THINC + MUSCL2, donor-cell eta          -> is consistent eta the amplifier?
# Submit:  sbatch --export=CASE=halfz RunSch20CollapsingAxisJet.sh   (repeat for the other two)
# Analyse: python tests/FlowRayleighPlesset/reference/diagnose_axis_jets.py <out dirs> --labels ...

#SBATCH --job-name=AxisJetSch20
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

EXE=/home/ttryon/flames/bin/hydro2-3d-g++
DECK=/home/ttryon/flames/tests/FlowRayleighPlesset/Limiter_Sweep/Sch20_Collapsing_Neumann_Large_3D_MUSCL2_THINC_ETA
OUT=/mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Collapsing_Large_3D_AXISJET_${CASE}
COMMON=(stop_time=3.2e-4 plot_file=${OUT})

case "${CASE}" in
  halfz)
    EXTRA=("geometry.prob_lo=0.0 0.0 -2.0" "amr.n_cell=160 160 320" "refine_box.lo=0.0 0.0 -0.03"
           "eta.bc.type.zlo=neumann" "density.bc.type.zlo=neumann" "energy.bc.type.zlo=neumann"
           "momentum.bc.type.zlo=neumann neumann neumann") ;;
  nothinc)
    EXTRA=(Limiter.type=muscl2 eta_consistent_advect=1) ;;
  thincdonor)
    EXTRA=(Limiter.type=thinc Limiter.thinc_base=muscl2 eta_consistent_advect=0) ;;
  *) echo "set CASE=halfz|nothinc|thincdonor"; exit 1 ;;
esac

srun --mpi=pmi2 ${EXE} ${DECK} "${COMMON[@]}" "${EXTRA[@]}"
