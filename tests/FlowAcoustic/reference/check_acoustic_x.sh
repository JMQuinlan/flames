#!/bin/bash
# x-oriented diagnostic: NSCBC on x-faces (pre-existing code) vs x free-space
# truth, along the x-axis. Compare with the z result to localize the bug.
cd "$(dirname "$0")/../../.." || exit 1
FX=ext/AMReX-Codes/amrex/Tools/Plotfile/fextract.3d.ex
./configure --dim=3 --comp=g++ >/dev/null 2>&1
make -j12 bin/hydro2-3d-g++ >/tmp/b3.log 2>&1
for tag in NSCBCx Tallx; do
  echo "=== run $tag ==="
  mpirun -np 6 ./bin/hydro2-3d-g++ "tests/FlowAcoustic/input_AcousticPulse3D_${tag}" >/tmp/ap_$tag.log 2>&1
  grep -iE 'finalized|ABORT|nan' /tmp/ap_$tag.log | tail -1
done
last () { ls -d "$1"/[0-9]*cell 2>/dev/null | sort | tail -1; }
$FX -d 0 -y 0.0 -z 0.0 -v pressure -s /tmp/px_n.txt "$(last tests/FlowAcoustic/output_Pulse3D_NSCBCx)" 2>/dev/null
$FX -d 0 -y 0.0 -z 0.0 -v pressure -s /tmp/px_t.txt "$(last tests/FlowAcoustic/output_Pulse3D_Tallx)"  2>/dev/null
awk '!/^#/ && NF>=2 { z=$1+0; if (z>=-0.5 && z<=0.5){ printf "%.4f %s\n", z, $2 } }' /tmp/px_t.txt > /tmp/truthx.txt
awk -v tf=/tmp/truthx.txt 'BEGIN{ while((getline l<tf)>0){split(l,a," ");T[a[1]]=a[2]} }
  !/^#/ && NF>=2 { k=sprintf("%.4f",$1+0); if(k in T){d=$2-T[k]; if(d<0)d=-d; if(d>m)m=d} }
  END{ printf "  max|x-NSCBC - x-Truth| over x in [-0.5,0.5] = %.5g\n", m }' /tmp/px_n.txt
echo "  (compare to z-NSCBC vs z-Truth = 7.5e-4)"
