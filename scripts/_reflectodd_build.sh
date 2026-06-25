#!/bin/bash
cd /mnt/c/Users/tryon/Documents/flames || exit 1
echo "=== configure 3D + FULL build (make, all targets -- not just hydro2) ==="
./configure --dim=3 --comp=g++ > /tmp/rofix_cfg.log 2>&1
# touch all sources to defeat /mnt/c 9p mtime skew -> force a real recompile
find src -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.H' \) -exec touch {} +
make -j8 > /tmp/rofix_build.log 2>&1
BUILD_RC=$?
echo "FULL make rc=$BUILD_RC   error_lines=$(grep -ciE 'error:' /tmp/rofix_build.log)"
echo "--- 3D executables produced ---"
ls -1 bin/*-3d-g++ 2>/dev/null
echo "--- compile/link errors (if any) ---"
grep -iE 'error:|Error [0-9]|undefined reference' /tmp/rofix_build.log | grep -viE 'warning' | head -25
echo

if [ ! -f bin/hydro2-3d-g++ ]; then echo "NO hydro2 BINARY -- aborting test"; echo "ROFIXDONE"; exit 1; fi

echo "=== re-run octant verification pair ==="
run() {
  rm -rf "tests/FlowRayleighPlesset/$2"
  mpirun -np 6 ./bin/hydro2-3d-g++ "tests/FlowRayleighPlesset/$1" > "/tmp/$2.log" 2>&1
  echo "  $1 -> exit=$?  last=$(ls -d tests/FlowRayleighPlesset/$2/*cell 2>/dev/null | tail -1)  t=$(grep -oE 'TIME = [0-9.e+-]+' /tmp/$2.log | tail -1)"
}
run _OctVerify_Full_3D output_OctVerify_Full
run _OctVerify_Oct_3D  output_OctVerify_Oct
echo "ROFIXDONE"
