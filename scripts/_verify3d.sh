#!/bin/bash
cd /mnt/c/Users/tryon/Documents/flames || exit 1
echo "=== rebuild (touch defeats /mnt/c mtime skew) ==="
find src -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.H' \) -exec touch {} +
make -j8 bin/hydro2-3d-g++ > /tmp/v_build.log 2>&1
echo "build rc=$? errors=$(grep -ciE 'error:' /tmp/v_build.log)"
grep -iE "error:" /tmp/v_build.log | grep -viE "warning" | head -3
echo "=== smoke (max_step=5) ==="
declare -a T=(
  "FlowCouette/UNIT_TEST_3D_single"
  "FlowRiemannUnitTests/input_Garrick_z"
  "FlowLaplace/UNIT_TEST_3D_R1.0e-3_Sigma7.3e-2"
  "FlowRayleighPlesset/Sch20_Oscillating_Neumann_3D"
  "FlowRayleighPlesset/Sch20_Oscillating_NSCBC_3D"
  "FlowRayleighPlesset/Sch20_Collapsing_Neumann_3D"
  "FlowRayleighPlesset/Sch20_Collapsing_NSCBC_3D"
)
for t in "${T[@]}"; do
  n=$(basename "$t")
  mpirun -np 6 ./bin/hydro2-3d-g++ "tests/$t" max_step=5 plot_file=./tmp_v > /tmp/v_$n.log 2>&1
  if grep -qiE "ABORT|EXCEPTION|ERROR IN|Segfault|Assertion|errorcode" /tmp/v_$n.log; then
    echo "FAIL  $n :  $(grep -iE 'ABORT|ERROR IN|Assertion|errorcode|SIGABRT|SIGSEGV' /tmp/v_$n.log | head -1 | cut -c1-72)"
  else
    echo "OK    $n :  $(grep -iE 'STEP 5|finalized' /tmp/v_$n.log | tail -1 | cut -c1-40)"
  fi
  rm -rf ./tmp_v
done
