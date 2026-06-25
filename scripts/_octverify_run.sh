#!/bin/bash
cd /mnt/c/Users/tryon/Documents/flames || exit 1
echo "=== reconfigure 3D + rebuild (touch defeats /mnt/c mtime skew) ==="
./configure --dim=3 --comp=g++ > /tmp/ovcfg.log 2>&1
find src -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.H' \) -exec touch {} +
make -j8 bin/hydro2-3d-g++ > /tmp/ovbuild.log 2>&1
echo "build rc=$? errors=$(grep -ciE 'error:' /tmp/ovbuild.log)"
grep -iE 'error:' /tmp/ovbuild.log | grep -viE warning | head -5
[ -f bin/hydro2-3d-g++ ] || { echo "NO BINARY"; exit 1; }
echo "reflect_even compiled in: $(strings bin/hydro2-3d-g++ | grep -ic REFLECT_EVEN)"

run() {  # $1=input  $2=outdir
  rm -rf "tests/FlowRayleighPlesset/$2"
  mpirun -np 6 ./bin/hydro2-3d-g++ "tests/FlowRayleighPlesset/$1" > "/tmp/$2.log" 2>&1
  local rc=$?
  echo "  $1 -> exit=$rc  last=$(ls -d tests/FlowRayleighPlesset/$2/*cell 2>/dev/null | tail -1)  t=$(grep -oE 'TIME = [0-9.e+-]+' /tmp/$2.log | tail -1)"
  [ $rc -ne 0 ] && { echo "  --- tail of /tmp/$2.log ---"; tail -15 "/tmp/$2.log"; }
}
echo "=== run FULL (32^3, neumann) ==="
run _OctVerify_Full_3D output_OctVerify_Full
echo "=== run OCTANT (16^3, symmetry) ==="
run _OctVerify_Oct_3D  output_OctVerify_Oct
echo "OVDONE"
