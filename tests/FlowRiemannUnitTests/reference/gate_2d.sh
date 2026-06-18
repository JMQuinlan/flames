#!/bin/bash
# 2D regression gate: configure 2D, build, run Garrick, fcompare vs golden.
# Run from repo root.  Prints only the build result, run status, and any
# fcompare rows whose error is NOT exactly zero (kappaAvg ~3e-19 is expected).
cd "$(dirname "$0")/../../.." || exit 1   # -> repo root
./configure --dim=2 --comp=g++ >/dev/null 2>&1
echo "=== build 2D ==="
make -j12 bin/hydro2-2d-g++ > /tmp/build2d.log 2>&1
if grep -qiE 'error:|No rule to make' /tmp/build2d.log; then
  echo "BUILD FAILED:"; grep -iE 'error:|No rule' /tmp/build2d.log | grep -viE 'warning' | head; exit 1
fi
grep -i LINKING /tmp/build2d.log | tail -1
echo "=== run Garrick (6 ranks) ==="
mpirun -np 6 ./bin/hydro2-2d-g++ tests/FlowRiemannUnitTests/input_Garrick > /tmp/g.log 2>&1
tail -1 /tmp/g.log
echo "=== fcompare vs golden (rows with nonzero error) ==="
ext/AMReX-Codes/amrex/Tools/Plotfile/fcompare.2d.ex \
    tests/FlowRiemannUnitTests/output_Garrick_GOLDEN_2D \
    tests/FlowRiemannUnitTests/output_Garrick/02000cell 2>/dev/null \
  | grep -vE '[[:space:]]0[[:space:]]+0$'
