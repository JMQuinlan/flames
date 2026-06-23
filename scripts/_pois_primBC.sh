#!/bin/bash
cd /mnt/c/Users/tryon/Documents/flames || exit 1
echo "=== reconfigure 2D + rebuild (touch defeats /mnt/c mtime skew) ==="
./configure --dim=2 --comp=g++ > /tmp/cfg2.log 2>&1
find src -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.H' \) -exec touch {} +
make -j8 bin/hydro2-2d-g++ > /tmp/b2d.log 2>&1
echo "build rc=$? errors=$(grep -ciE 'error:' /tmp/b2d.log)"
grep -iE 'error:' /tmp/b2d.log | grep -viE warning | head -3
[ -f bin/hydro2-2d-g++ ] || { echo "NO BINARY"; exit 1; }
echo "primitive flag compiled in: $(strings bin/hydro2-2d-g++ | grep -c 'bc.primitive')"

run() {  # $1=input  $2=outdir
  rm -rf "$2"
  mpirun -np 6 ./bin/hydro2-2d-g++ "tests/FlowPoseuille/$1" \
      stop_time=12.0 plot_file="$2" amr.plot_int=4000 > "/tmp/$2.log" 2>&1
  echo "  $1 -> exit=$? last=$(ls -d $2/*cell 2>/dev/null | tail -1)"
}
echo "=== run ENERGY BC (original) ==="
run UNIT_TEST_2D    tmp_eng
echo "=== run PRIMITIVE pressure BC ==="
run _smoke_primBC_2D tmp_prim
echo "DONE"
