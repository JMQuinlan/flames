#!/bin/bash
# 10-case Marmottant Laplace sweep, NSCBC outflow only.  No fallback.
set -u
TMPL=/home/ttryon/Desktop/flames2/tests/FlowMarmottant/TEMPLATES
GEN=/home/ttryon/Desktop/flames2/tests/FlowMarmottant/GENERATED_nscbc_sep8
BIN=/home/ttryon/Desktop/flames2/bin
OUT=$BIN/tests/FlowMarmottant/nscbc_sep8
TAG=nscbc_sep8

cd $TMPL
rm -rf $GEN $OUT
echo "### generating 10 NSCBC inputs"
DRYRUN=1 TAG=$TAG BCMODE=nscbc ./RUN_MARMOTTANT_SWEEP.bash 2>&1 | grep -E "ratio=|R_buck|domain|stop_time"
for f in $GEN/input_R*; do
  n=$(grep -c '@[A-Z_]*@' "$f"); [ "$n" -ne 0 ] && { echo "FATAL: $n unfilled tokens in $f"; exit 1; }
done
grep -q nscbc_outflow $GEN/input_R001000000 || { echo "FATAL: NSCBC block missing"; exit 1; }
echo "### token check passed; nscbc_outflow confirmed in inputs"

mkdir -p $OUT/logs
cd $BIN
echo; echo "### running all 10 concurrently, 3 ranks each (30/32 cores)"
for inp in $GEN/input_R*; do
  name=$(basename $inp | sed 's/input_//')
  ( timeout 21600 mpirun -np 3 ./hydro2-2d-g++ "$inp" > $OUT/logs/$name.log 2>&1 ) &
done
wait
echo; echo "### results"
for inp in $GEN/input_R*; do
  name=$(basename $inp | sed 's/input_//')
  L=$OUT/logs/$name.log
  a=$(grep -c "MPI_ABORT\|Couldn.t open\|assertion" $L 2>/dev/null)
  t=$(grep -oP 'Time = \K[0-9.eE+-]+' $L 2>/dev/null | tail -1)
  p=$(ls -d $OUT/$name/*cell 2>/dev/null | wc -l)
  printf "    %-12s aborts=%-3s t=%-16s plots=%s\n" "$name" "$a" "${t:-none}" "$p"
done

echo; echo "### analysis"
cd /home/ttryon/Desktop/flames2/tests/FlowMarmottant/reference
MBDIR=$OUT RB=1.0e-3 RSCALE=1e3 RLABEL='R_0 (mm)' RUNIT=mm \
 TSCALE=1e3 TLABEL='Time (ms)' TUNIT=ms TEND=1.0e-2 FIGNAME=fig_nscbc_sep8 \
 python3 microbubble_fig.py 2>&1 | tail -45
echo "NSCBC_SWEEP_COMPLETE"
