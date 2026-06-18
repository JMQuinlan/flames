#!/bin/bash
# Build 3D, run the all-open-box NSCBC test, report crash/finalize status and
# the final-time pressure residual max|p-1| (should be small if non-reflecting).
cd "$(dirname "$0")/../../.." || exit 1
FE=ext/AMReX-Codes/amrex/Tools/Plotfile/fextrema.3d.ex
./configure --dim=3 --comp=g++ >/dev/null 2>&1
make -j12 bin/hydro2-3d-g++ >/tmp/b3.log 2>&1
grep -qiE 'error:' /tmp/b3.log && { echo BUILD FAIL; grep -i error: /tmp/b3.log | head; exit 1; }
rm -f Backtrace.* 2>/dev/null
mpirun -np 6 ./bin/hydro2-3d-g++ tests/FlowAcoustic/input_AcousticPulse3D_Open > /tmp/open.log 2>&1
echo "--- status ---"
grep -iE 'finalized|ABORT|segfault|nan|corrupted' /tmp/open.log | tail -3
last=$(ls -d tests/FlowAcoustic/output_Pulse3D_Open/[0-9]*cell 2>/dev/null | sort | tail -1)
echo "--- final plotfile: $last ---"
$FE -v pressure "$last" 2>/dev/null | tail -3
