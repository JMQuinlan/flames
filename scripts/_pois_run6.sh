#!/bin/bash
pkill -f hydro2-2d-g++ 2>/dev/null; sleep 1
cd /mnt/c/Users/tryon/Documents/flames || exit 1
run() {  # $1=input $2=outdir
  rm -rf "$2"
  mpirun -np 6 ./bin/hydro2-2d-g++ "tests/FlowPoseuille/$1" \
      stop_time=6.0 plot_file="$2" amr.plot_int=100000 > "/tmp/$2.log" 2>&1
  echo "  $1 -> exit=$? t=$(grep -oE 'TIME = [0-9.e+-]+' /tmp/$2.log | tail -1)  last=$(ls -d $2/*cell 2>/dev/null | tail -1)"
}
echo "=== ENERGY BC -> t=6 ==="
run UNIT_TEST_2D     tmp_eng
echo "=== PRIMITIVE BC -> t=6 ==="
run _smoke_primBC_2D tmp_prim
echo "RUNDONE"
