#!/bin/bash
# 3D gate: configure 3D, build, smoke-run Garrick_x and Garrick_z, then validate
# transverse uniformity + x/z rotational invariance.  Run from anywhere.
# Arg 1 = stop_time override (default 0.01 = ~100 steps smoke).
cd "$(dirname "$0")/../../.." || exit 1   # -> repo root
ST=${1:-0.01}
./configure --dim=3 --comp=g++ >/dev/null 2>&1
echo "=== build 3D ==="
make -j12 bin/hydro2-3d-g++ > /tmp/build3d.log 2>&1
if grep -qiE 'error:|No rule to make' /tmp/build3d.log; then
  echo "BUILD FAILED:"; grep -iE 'error:|No rule' /tmp/build3d.log | grep -viE 'warning' | head; exit 1
fi
grep -i LINKING /tmp/build3d.log | tail -1
for d in x z; do
  echo "=== run Garrick_$d (stop_time=$ST) ==="
  mpirun -np 6 ./bin/hydro2-3d-g++ "tests/FlowRiemannUnitTests/input_Garrick_$d" stop_time=$ST > /tmp/g3$d.log 2>&1
  grep -iE 'finalized|ABORT|nan' /tmp/g3$d.log | tail -1
done
echo "=== validation ==="
bash tests/FlowRiemannUnitTests/reference/check_garrick3d.sh 2>/dev/null \
  | grep -vE '^ slicing'
