#!/bin/bash
cd /mnt/c/Users/tryon/Documents/flames || exit 1
echo "=== configure 2D + build (Marmottant code) ==="
./configure --dim=2 --comp=g++ > /tmp/marm_cfg.log 2>&1
find src -type f \( -name '*.cpp' -o -name '*.H' \) -exec touch {} +
make -j8 bin/hydro2-2d-g++ > /tmp/marm_build.log 2>&1
echo "build rc=$? errors=$(grep -ciE 'error:' /tmp/marm_build.log)"
grep -iE 'error:' /tmp/marm_build.log | grep -viE warning | head -8
[ -f bin/hydro2-2d-g++ ] || { echo "NO BINARY"; exit 1; }
echo "marmottant compiled in: $(strings bin/hydro2-2d-g++ | grep -c 'Marmottant')"

echo "=== run 2D Laplace Marmottant ==="
rm -rf tests/FlowMarmottant/output_Laplace_Marmottant_2D
mpirun -np 6 ./bin/hydro2-2d-g++ tests/FlowMarmottant/input_Laplace_Marmottant_2D \
    > /tmp/marm_2d.log 2>&1
echo "  exit=$?  frames=$(ls -d tests/FlowMarmottant/output_Laplace_Marmottant_2D/*cell 2>/dev/null | wc -l)  t=$(grep -oE 'TIME = [0-9.e+-]+' /tmp/marm_2d.log | tail -1)"
echo "--- first + last Marmottant R / sigma_eff reported by solver ---"
grep -oE "Marmottant lev=0 R=[0-9.e+-]+ Rb=[0-9.e+-]+ sigma_eff=[0-9.e+-]+" /tmp/marm_2d.log | head -1
grep -oE "Marmottant lev=0 R=[0-9.e+-]+ Rb=[0-9.e+-]+ sigma_eff=[0-9.e+-]+" /tmp/marm_2d.log | tail -1
echo "--- any abort/error? ---"
grep -iE "abort|SIGABRT|error|nan" /tmp/marm_2d.log | grep -viE "error check|ERROR IN" | head -5
echo "MARMDONE"
