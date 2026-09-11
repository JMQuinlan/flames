#!/bin/bash
# 10-radius Marmottant Laplace sweep -- FULL DOMAIN, DIRICHLET p + momentum.
# No NSCBC anywhere.  4 cases in flight x 7 ranks = 28 cores.  No time limit.
set -u
GEN=/home/ttryon/Desktop/flames2/tests/FlowMarmottant/GENERATED_dirich
BIN=/home/ttryon/Desktop/flames2/bin
OUT=$BIN/tests/FlowMarmottant/dirich
NP=7
NPAR=4
mkdir -p $OUT/logs
for f in $GEN/input_R*; do
  n=$(grep -c '@[A-Z_]*@' "$f")
  [ "$n" -ne 0 ] && { echo "FATAL: $n unfilled tokens in $f"; exit 1; }
  grep -q "nscbc" "$f" && { echo "FATAL: nscbc leaked into $f"; exit 1; }
  grep -q "^bc.primitive = 1" "$f" || { echo "FATAL: bc.primitive not set: $f"; exit 1; }
  grep -q "^pressure.bc.type.xlo = dirichlet" "$f" || { echo "FATAL: pressure not dirichlet: $f"; exit 1; }
  grep -q "^momentum.bc.type.xlo = dirichlet dirichlet" "$f" || { echo "FATAL: momentum not dirichlet: $f"; exit 1; }
  grep -q "^stop_time = 1.0e-2" "$f" || { echo "FATAL: wrong stop_time: $f"; exit 1; }
  grep -q "^omega_refinement_criterion = 1.0e+10" "$f" || { echo "FATAL: omega refinement active: $f"; exit 1; }
done
echo "### preflight OK: 10/10 full-domain, dirichlet p+momentum, eta-only, 10 ms, no nscbc"
cd $BIN
date '+### start %Y-%m-%d %H:%M:%S'
run_one() {
    local inp=$1
    local name=$(basename "$inp" | sed 's/input_//')
    mpirun -np $NP --mca mpi_yield_when_idle 1 --bind-to none ./hydro2-2d-g++ "$inp" > $OUT/logs/$name.log 2>&1
    local t=$(grep -oP 'TIME = \K[0-9.eE+-]+' $OUT/logs/$name.log | tail -1)
    local a=$(grep -c "MPI_ABORT\|assertion" $OUT/logs/$name.log)
    echo "### DONE $name t=${t:-none} aborts=$a $(date '+%H:%M:%S')"
}
for inp in $GEN/input_R*; do
    while [ "$(jobs -rp | wc -l)" -ge $NPAR ]; do wait -n; done
    run_one "$inp" &
done
wait
echo "### results"
for inp in $GEN/input_R*; do
    n=$(basename "$inp" | sed 's/input_//')
    L=$OUT/logs/$n.log
    printf "    %-12s aborts=%-3s t=%-16s plots=%s\n" "$n" \
      "$(grep -c 'MPI_ABORT\|assertion' $L 2>/dev/null)" \
      "$(grep -oP 'TIME = \K[0-9.eE+-]+' $L 2>/dev/null | tail -1)" \
      "$(ls -d $OUT/$n/*cell 2>/dev/null | wc -l)"
done
echo DIRICH_COMPLETE
