#!/bin/bash
# 3D oblique acoustic-pulse non-reflection metric.
# Builds 3D, runs the NSCBC and the (reflecting) Wall variants, and reports the
# residual pressure deviation max|p - p_base| at the final time for each.
# A working non-reflecting BC => NSCBC residual << Wall residual.
cd "$(dirname "$0")/../../.." || exit 1
FX=ext/AMReX-Codes/amrex/Tools/Plotfile/fextrema.3d.ex

./configure --dim=3 --comp=g++ >/dev/null 2>&1
echo "=== build 3D ==="
make -j12 bin/hydro2-3d-g++ > /tmp/b3.log 2>&1
if grep -qiE 'error:' /tmp/b3.log; then echo "BUILD FAILED"; grep "error:" /tmp/b3.log | head; exit 1; fi
grep -i LINKING /tmp/b3.log | tail -1

resid () { # $1 = plotfile dir -> prints max|p-1|
  $FX "$1" 2>/dev/null | awk '/^ pressure/{mn=$2; mx=$3; d1=(mx>1)?mx-1:1-mx; d2=(mn>1)?mn-1:1-mn; printf "%.5g", (d1>d2)?d1:d2}'
}

for tag in NSCBC Wall; do
  echo "=== run $tag ==="
  mpirun -np 6 ./bin/hydro2-3d-g++ "tests/FlowAcoustic/input_AcousticPulse3D_${tag}" > /tmp/ap_$tag.log 2>&1
  grep -iE 'finalized|ABORT|nan' /tmp/ap_$tag.log | tail -1
  OUT=$(ls -d tests/FlowAcoustic/output_Pulse3D_${tag}/[0-9]*cell 2>/dev/null | sort | tail -1)
  echo "  final = $OUT ; residual max|p-1| = $(resid "$OUT")"
done
