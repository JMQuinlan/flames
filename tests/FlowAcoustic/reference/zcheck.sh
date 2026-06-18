#!/bin/bash
# Lean z-NSCBC vs existing z-truth check (reuses output_Pulse3D_Tall).
cd "$(dirname "$0")/../../.." || exit 1
FX=ext/AMReX-Codes/amrex/Tools/Plotfile/fextract.3d.ex
./configure --dim=3 --comp=g++ >/dev/null 2>&1
make -j12 bin/hydro2-3d-g++ >/tmp/b3.log 2>&1
grep -qiE 'error:' /tmp/b3.log && { echo BUILD FAIL; grep error: /tmp/b3.log|head; exit 1; }
mpirun -np 6 ./bin/hydro2-3d-g++ tests/FlowAcoustic/input_AcousticPulse3D_NSCBC >/tmp/apz.log 2>&1
grep -iE 'finalized|ABORT|nan' /tmp/apz.log | tail -1
last () { ls -d "$1"/[0-9]*cell 2>/dev/null | sort | tail -1; }
$FX -d 2 -x 0.0 -y 0.0 -v pressure -s /tmp/pz_n.txt "$(last tests/FlowAcoustic/output_Pulse3D_NSCBC)" 2>/dev/null
$FX -d 2 -x 0.0 -y 0.0 -v pressure -s /tmp/pz_t.txt "$(last tests/FlowAcoustic/output_Pulse3D_Tall)"  2>/dev/null
awk '!/^#/ && NF>=2 { z=$1+0; if(z>=-0.5&&z<=0.5) printf "%.4f %s\n",z,$2 }' /tmp/pz_t.txt > /tmp/tz.txt
awk -v tf=/tmp/tz.txt 'BEGIN{while((getline l<tf)>0){split(l,a," ");T[a[1]]=a[2]}}
  !/^#/&&NF>=2{k=sprintf("%.4f",$1+0); if(k in T){d=$2-T[k];if(d<0)d=-d;if(d>m)m=d}}
  END{printf "  max|z-NSCBC - z-Truth| = %.5g  (was 7.5e-4 with 1b+2 enabled; x-NSCBC = 1.5e-5)\n", m}' /tmp/pz_n.txt
