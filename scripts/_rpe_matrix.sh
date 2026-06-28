#!/bin/bash
cd /mnt/c/Users/tryon/Documents/flames || exit 1
echo "=== build 3D (robust fallback + Emix dump) ==="
./configure --dim=3 --comp=g++ > /tmp/mtx_cfg.log 2>&1
find src/Integrator/Hydro2.cpp -exec touch {} +
make -j8 bin/hydro2-3d-g++ > /tmp/mtx_build.log 2>&1
echo "build rc=$? errors=$(grep -ciE 'error:' /tmp/mtx_build.log)"
grep -iE 'error:' /tmp/mtx_build.log | grep -viE warning | head -6
[ -f bin/hydro2-3d-g++ ] || { echo NO BINARY; exit 1; }

run () {
  local name=$1
  local log=/tmp/mtx_${name}.log
  rm -rf tests/FlowRayleighPlesset/output_smoke_RPE_${name}_3D
  mpirun -np 6 ./bin/hydro2-3d-g++ tests/FlowRayleighPlesset/smoke_RPE_${name}_3D \
        plot_file=./tests/FlowRayleighPlesset/output_smoke_RPE_${name}_3D > "$log" 2>&1
  local ec=$?
  local steps=$(grep -c "TIME =" "$log")
  local rf=$(grep -c "RELAX-FAIL" "$log")
  local nfail=$(grep -E "unconverged_cells=[1-9]" "$log" | wc -l)
  local ncall=$(grep -c "unconverged_cells=" "$log")
  printf "  %-14s exit=%s steps=%-3s RELAX-FAIL=%-4s relax_fail_calls=%s/%s\n" "$name" "$ec" "$steps" "$rf" "$nfail" "$ncall"
  # show the worst-step unconverged + first Emix (is the conserved E sane?)
  local emix=$(grep "RELAX-FAIL" "$log" | head -1 | grep -oE "Emix=[-0-9.e+]+")
  [ -n "$emix" ] && echo "                 first-fail $emix  (E0+E1<0 => garbage ; ~+4e4 => sane)"
}
echo "=== RESULTS (exit=6 means CRASH) ==="
run octant
run full
run octant_coarse
run full_coarse
run octant_amr
echo MTXDONE
