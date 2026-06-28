#!/bin/bash
cd /mnt/c/Users/tryon/Documents/flames || exit 1
echo "=== configure 3D + build (RELAX-FAIL instrumentation) ==="
./configure --dim=3 --comp=g++ > /tmp/rpeab_cfg.log 2>&1
find src/Integrator/Hydro2.cpp -exec touch {} +
make -j8 bin/hydro2-3d-g++ > /tmp/rpeab_build.log 2>&1
echo "build rc=$? errors=$(grep -ciE 'error:' /tmp/rpeab_build.log)"
grep -iE 'error:' /tmp/rpeab_build.log | grep -viE warning | head -6
[ -f bin/hydro2-3d-g++ ] || { echo NO BINARY; exit 1; }

run_and_report () {
  local name=$1
  local log=/tmp/rpeab_${name}.log
  echo
  echo "############################################################"
  echo "### RUN: $name"
  echo "############################################################"
  rm -rf tests/FlowRayleighPlesset/output_smoke_RPE_${name}_3D
  mpirun -np 6 ./bin/hydro2-3d-g++ tests/FlowRayleighPlesset/smoke_RPE_${name}_3D > "$log" 2>&1
  echo "  exit=$?  (steps run ~ $(grep -c 'TIME =' "$log"))"
  echo "  --- RELAX-FAIL cells (first 6) ---"
  grep "RELAX-FAIL" "$log" | head -6
  echo "  RELAX-FAIL total: $(grep -c 'RELAX-FAIL' "$log")"
  echo "  --- relax_diag: unconverged>0 calls / total, worst ---"
  python3 - "$log" <<'PY'
import re,sys
rx=re.compile(r"unconverged_cells=(\d+)")
u=[int(m.group(1)) for line in open(sys.argv[1],errors='ignore') for m in [rx.search(line)] if m]
if u:
    bad=sum(1 for x in u if x>0)
    print(f"     {bad}/{len(u)} relax calls had unconverged_cells>0 ; max={max(u)}")
else:
    print("     no relax lines parsed")
PY
}

run_and_report octant
run_and_report full
echo
echo "RPEABDONE"
