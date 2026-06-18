#!/bin/bash
# 2D NSCBC4 regression gate. Builds 2D, runs a short input_Garrick_NSCBC, and
# (if a golden exists) fcompares the final state. With arg "golden" it instead
# SAVES the current final state as the golden baseline.
# Usage: bash gate_nscbc_2d.sh [golden]
cd "$(dirname "$0")/../../.." || exit 1
MODE=${1:-check}
ST=0.05   # ~500 steps; long enough for waves to reach the NSCBC boundary
GOLD=tests/FlowRiemannUnitTests/output_GarrickNSCBC_GOLDEN_2D
OUT=tests/FlowRiemannUnitTests/output_Garrick   # plot_file set in input

./configure --dim=2 --comp=g++ >/dev/null 2>&1
echo "=== build 2D ==="
make -j12 bin/hydro2-2d-g++ > /tmp/bnscbc.log 2>&1
if grep -qiE 'error:|No rule to make' /tmp/bnscbc.log; then
  echo "BUILD FAILED:"; grep -iE 'error:|No rule' /tmp/bnscbc.log | grep -viE 'warning' | head; exit 1
fi
grep -i LINKING /tmp/bnscbc.log | tail -1
echo "=== run input_Garrick_NSCBC (stop_time=$ST) ==="
mpirun -np 6 ./bin/hydro2-2d-g++ tests/FlowRiemannUnitTests/input_Garrick_NSCBC stop_time=$ST > /tmp/gn.log 2>&1
grep -iE 'finalized|ABORT|nan' /tmp/gn.log | tail -1
# Final cell dir = highest-numbered NNNNNcell
FINAL=$(ls -d ${OUT}/[0-9]*cell 2>/dev/null | sort | tail -1)
echo "final state: $FINAL"
if [ "$MODE" = golden ]; then
  rm -rf "$GOLD"; cp -r "$FINAL" "$GOLD" && echo "GOLDEN saved to $GOLD"
else
  echo "=== fcompare vs golden (nonzero-error rows) ==="
  ext/AMReX-Codes/amrex/Tools/Plotfile/fcompare.2d.ex "$GOLD" "$FINAL" 2>/dev/null \
    | grep -vE '[[:space:]]0[[:space:]]+0$'
fi
