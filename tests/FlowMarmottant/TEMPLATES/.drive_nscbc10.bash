#!/bin/bash
# =============================================================================
# Marmottant 10-radius Laplace sweep -- FINAL CONFIG
#   NSCBC outflow only (no Neumann fallback, no BC auto-select)
#   eta-only refinement (omega/p/rho/gradu all 1e10)
#   stop_time = 1.0e-2 s (10 ms)
#   NO time limit on any run
#   4 cases in flight x 8 MPI ranks = 32 cores
# =============================================================================
set -u
TMPL=/home/ttryon/Desktop/flames2/tests/FlowMarmottant/TEMPLATES
GEN=/home/ttryon/Desktop/flames2/tests/FlowMarmottant/GENERATED_nscbc10
BIN=/home/ttryon/Desktop/flames2/bin
OUT=$BIN/tests/FlowMarmottant/nscbc10
NP=7          # ranks per case (4x7=28 +4 orted = 32, no oversubscription)
NPAR=4        # cases in flight

cd $TMPL
rm -rf $GEN $OUT
mkdir -p $OUT/logs

echo "### generating 10 NSCBC inputs (stop_time=1.0e-2)"
DRYRUN=1 TAG=nscbc10 BCMODE=nscbc STOP=1.0e-2 ./RUN_MARMOTTANT_SWEEP.bash 2>&1 \
  | grep -E "ratio=|R_buck|R_rupture|domain|capillary"

# ---- hard preflight: fail loudly rather than silently running the wrong thing
for f in $GEN/input_R*; do
  n=$(grep -c '@[A-Z_]*@' "$f")
  [ "$n" -ne 0 ] && { echo "FATAL: $n unfilled tokens in $f"; exit 1; }
  grep -q "nscbc_outflow" "$f"    || { echo "FATAL: not NSCBC: $f"; exit 1; }
  grep -q "neumann"      "$f"     && { echo "FATAL: neumann leaked in: $f"; exit 1; }
  grep -q "^omega_refinement_criterion = 1.0e+10" "$f" || { echo "FATAL: omega refinement active: $f"; exit 1; }
  grep -q "^stop_time = 1.0e-2"   "$f" || { echo "FATAL: wrong stop_time: $f"; exit 1; }
done
echo "### preflight OK: 10/10 inputs are NSCBC + eta-only + 10 ms, no unfilled tokens"

cd $BIN
echo; echo "### running: $NPAR cases in flight x $NP ranks = $((NPAR*NP)) cores, NO time limit"
date '+### start %Y-%m-%d %H:%M:%S'

run_one() {
    local inp=$1
    local name=$(basename "$inp" | sed 's/input_//')
    mpirun -np $NP --mca mpi_yield_when_idle 1 --bind-to none ./hydro2-2d-g++ "$inp" > $OUT/logs/$name.log 2>&1
    local t=$(grep -oP 'TIME = \K[0-9.eE+-]+' $OUT/logs/$name.log | tail -1)
    local a=$(grep -c "MPI_ABORT\|assertion" $OUT/logs/$name.log)
    echo "### DONE $name  t=${t:-none}  aborts=$a  $(date '+%H:%M:%S')"
}

for inp in $GEN/input_R*; do
    while [ "$(jobs -rp | wc -l)" -ge $NPAR ]; do wait -n; done
    run_one "$inp" &
done
wait
date '+### all runs ended %Y-%m-%d %H:%M:%S'

echo; echo "### results"
printf "    %-12s %-8s %-18s %s\n" case aborts t_reached plots
for inp in $GEN/input_R*; do
    name=$(basename "$inp" | sed 's/input_//')
    L=$OUT/logs/$name.log
    a=$(grep -c "MPI_ABORT\|assertion" $L 2>/dev/null)
    t=$(grep -oP 'TIME = \K[0-9.eE+-]+' $L 2>/dev/null | tail -1)
    p=$(ls -d $OUT/$name/*cell 2>/dev/null | wc -l)
    printf "    %-12s %-8s %-18s %s\n" "$name" "$a" "${t:-none}" "$p"
done

echo; echo "### analysis"
cd /home/ttryon/Desktop/flames2/tests/FlowMarmottant/reference
MBDIR=$OUT RB=1.0e-3 RSCALE=1e3 RLABEL='$R_0$  (mm)' RUNIT=mm TAGSCALE=1e-9 \
 TSCALE=1e3 TLABEL='Time  (ms)' TUNIT=ms TEND=1.0e-2 SIGW=0.073 CHI=0.55 \
 FIGNAME=fig_nscbc10 FIG_OUT=/home/ttryon/Downloads/Diffuse_Marmottant_Shell \
 python3 microbubble_fig.py 2>&1 | tail -50
echo "NSCBC10_COMPLETE"
