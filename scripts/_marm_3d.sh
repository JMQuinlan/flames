#!/bin/bash
cd /mnt/c/Users/tryon/Documents/flames || exit 1
echo "=== configure 3D + build ==="
./configure --dim=3 --comp=g++ > /tmp/marm3_cfg.log 2>&1
find src -type f \( -name '*.cpp' -o -name '*.H' \) -exec touch {} +
make -j8 bin/hydro2-3d-g++ > /tmp/marm3_build.log 2>&1
echo "build rc=$? errors=$(grep -ciE 'error:' /tmp/marm3_build.log)"
grep -iE 'error:' /tmp/marm3_build.log | grep -viE warning | head -5
[ -f bin/hydro2-3d-g++ ] || { echo NO BINARY; exit 1; }
echo "=== run 3D Laplace Marmottant ==="
rm -rf tests/FlowMarmottant/output_Laplace_Marmottant_3D
mpirun -np 6 ./bin/hydro2-3d-g++ tests/FlowMarmottant/input_Laplace_Marmottant_3D > /tmp/marm_3d.log 2>&1
echo "  exit=$?  frames=$(ls -d tests/FlowMarmottant/output_Laplace_Marmottant_3D/*cell 2>/dev/null | wc -l)  t=$(grep -oE 'TIME = [0-9.e+-]+' /tmp/marm_3d.log | tail -1)"
grep -oE "Marmottant lev=0 R=[0-9.e+-]+ Rb=[0-9.e+-]+ sigma_eff=[0-9.e+-]+" /tmp/marm_3d.log | head -1
grep -oE "Marmottant lev=0 R=[0-9.e+-]+ Rb=[0-9.e+-]+ sigma_eff=[0-9.e+-]+" /tmp/marm_3d.log | tail -1
echo "MARM3DONE"
