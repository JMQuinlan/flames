#!/bin/bash
# Marmottant sweep driver: smoke-test NSCBC -> choose BC -> run 10 -> figure.
set -u
cd /home/ttryon/Desktop/flames2/tests/FlowMarmottant/TEMPLATES
BIN=/home/ttryon/Desktop/flames2/bin
OUTROOT=$BIN/tests/FlowMarmottant
TAG=${TAG:-sweep_sep8}
NCORE=$(nproc)

echo "### host: $NCORE cores; binary $(date -r $BIN/hydro2-2d-g++ '+%m-%d %H:%M')"

# ---------------- 1. smoke test NSCBC ------------------------------------
echo; echo "### [1/4] NSCBC smoke test (R/Rb=1.00, to t=1.2e-3; old abort was 5.38e-4)"
DRYRUN=1 TAG=${TAG}_smoke RATIOS="1.00" BCMODE=nscbc STOP=1.2e-3 \
  ./RUN_MARMOTTANT_SWEEP.bash >/dev/null 2>&1
SM=/home/ttryon/Desktop/flames2/tests/FlowMarmottant/GENERATED_${TAG}_smoke/input_R001000000
cd $BIN
timeout 1500 mpirun -np 8 ./hydro2-2d-g++ $SM > /tmp/smoke_nscbc.log 2>&1
A=$(grep -c "MPI_ABORT\|Abort\|assertion" /tmp/smoke_nscbc.log)
T=$(grep -oP 'Time = \K[0-9.eE+-]+' /tmp/smoke_nscbc.log | tail -1)
echo "    aborts=$A   t_reached=${T:-none} / 1.2e-3"
if [ "$A" -eq 0 ] && [ -n "${T:-}" ] && python3 -c "exit(0 if $T > 1.1e-3 else 1)"; then
    BC=nscbc; echo "    -> NSCBC IS STABLE. Using nscbc_outflow (conservative far field)."
else
    BC=neumann; echo "    -> NSCBC still fails. Falling back to neumann (leaks, see decay note)."
fi

# ---------------- 2. generate the 10 inputs ------------------------------
echo; echo "### [2/4] generating 10 inputs with BCMODE=$BC"
cd /home/ttryon/Desktop/flames2/tests/FlowMarmottant/TEMPLATES
rm -rf ../GENERATED_$TAG $OUTROOT/$TAG
DRYRUN=1 TAG=$TAG BCMODE=$BC ./RUN_MARMOTTANT_SWEEP.bash 2>&1 | grep -E "R_buck|domain|capillary|sigma"

# ---------------- 3. run them (true parallelism, not xargs -P) -----------
NP=4; NPAR=$(( NCORE / NP )); [ $NPAR -lt 1 ] && NPAR=1; [ $NPAR -gt 5 ] && NPAR=5
echo; echo "### [3/4] running 10 cases, $NPAR at a time x $NP ranks"
cd $BIN
mkdir -p $OUTROOT/$TAG/logs
run_one() {
    local inp=$1 name=$(basename $1 | sed 's/input_//')
    timeout 14400 mpirun -np $NP ./hydro2-2d-g++ "$inp" > $OUTROOT/$TAG/logs/$name.log 2>&1
    local a=$(grep -c "MPI_ABORT\|Abort\|assertion" $OUTROOT/$TAG/logs/$name.log)
    local t=$(grep -oP 'Time = \K[0-9.eE+-]+' $OUTROOT/$TAG/logs/$name.log | tail -1)
    local n=$(ls -d $OUTROOT/$TAG/$name/*cell 2>/dev/null | wc -l)
    printf "    %-12s aborts=%s  t=%-14s plots=%s\n" "$name" "$a" "${t:-none}" "$n"
}
i=0
for inp in /home/ttryon/Desktop/flames2/tests/FlowMarmottant/GENERATED_$TAG/input_R*; do
    run_one "$inp" &
    i=$((i+1)); [ $((i % NPAR)) -eq 0 ] && wait
done
wait
echo "    all cases finished"

# ---------------- 4. analysis --------------------------------------------
echo; echo "### [4/4] analysis"
cd /home/ttryon/Desktop/flames2/tests/FlowMarmottant/reference
MBDIR=$OUTROOT/$TAG RB=1.0e-3 RSCALE=1e3 RLABEL="R_0 (mm)" RUNIT=mm \
  TSCALE=1e3 TLABEL="Time (ms)" TUNIT=ms TEND=1.0e-2 \
  FIGNAME=fig_${TAG} BCMODE_NOTE=$BC python3 microbubble_fig.py 2>&1 | tail -40
echo; echo "BCMODE_USED=$BC"; echo "SWEEP_COMPLETE"
