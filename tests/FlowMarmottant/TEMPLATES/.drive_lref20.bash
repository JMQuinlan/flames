#!/bin/bash
# 10-radius Marmottant sweep with L_ref = domain/20 (was domain).
# NSCBC sigma=0.6, full domain, eta-only refinement, 10 ms, no time limit.
set -u
GEN=/home/ttryon/Desktop/flames2/tests/FlowMarmottant/GENERATED_lref20
BIN=/home/ttryon/Desktop/flames2/bin
OUT=$BIN/tests/FlowMarmottant/lref20
NP=7
NPAR=4
mkdir -p $OUT/logs
for f in $GEN/input_R*; do
  n=$(grep -c '@[A-Z_]*@' "$f")
  [ "$n" -ne 0 ] && { echo "FATAL: $n unfilled tokens in $f"; exit 1; }
  grep -q "^nscbc.xlo.L_ref = 2.500000e-04" "$f" || { echo "FATAL: L_ref not 2.5e-4 in $f"; exit 1; }
  grep -q "^nscbc.xlo.sigma = 0.6" "$f" || { echo "FATAL: sigma not 0.6 in $f"; exit 1; }
  grep -q "^stop_time = 1.0e-2" "$f" || { echo "FATAL: wrong stop_time: $f"; exit 1; }
  grep -q "^omega_refinement_criterion = 1.0e+10" "$f" || { echo "FATAL: omega refinement active: $f"; exit 1; }
  grep -q "neumann" "$f" && { echo "FATAL: neumann leaked into $f"; exit 1; }
done
echo "### preflight OK: 10/10 L_ref=2.5e-4 (domain/20), sigma=0.6, eta-only, 10 ms"
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
    n=$(basename "$inp" | sed 's/input_//'); L=$OUT/logs/$n.log
    printf "    %-12s aborts=%-3s t=%-16s plots=%s\n" "$n" \
      "$(grep -c 'MPI_ABORT\|assertion' $L 2>/dev/null)" \
      "$(grep -oP 'TIME = \K[0-9.eE+-]+' $L 2>/dev/null | tail -1)" \
      "$(ls -d $OUT/$n/[0-9]*cell 2>/dev/null | wc -l)"
done
# mb_* symlinks for microbubble_fig.py, then the normalized-radius analysis
cd $OUT
for d in R[0-9]*/; do n=${d%/}; num=$(echo ${n#R} | sed 's/^0*//'); [ -L "mb_$num" ] || ln -s "$n" "mb_$num"; done
echo; echo "### normalized radius analysis"
cd /home/ttryon/Desktop/flames2/tests/FlowMarmottant/reference
MBDIR=lref20 RB=1.0e-3 RSCALE=1e3 RLABEL='$R_0$  (mm)' RUNIT=mm TAGSCALE=1e-9 \
 TSCALE=1e3 TLABEL='Time  (ms)' TUNIT=ms TEND=1.0e-2 SIGW=0.073 CHI=0.55 \
 FIGNAME=fig_lref20 FIG_OUT=/home/ttryon/Desktop/flames2/tests/FlowMarmottant/reference \
 python3 microbubble_fig.py 2>&1 | grep -viE "warning|futurew" | tail -32
echo LREF20_COMPLETE
