#!/bin/bash
# Run the tall free-space truth and compare NSCBC and Wall z-axis pressure
# profiles against it over z in [-0.5,0.5].  Non-reflecting BC => |NSCBC-Truth|
# << |Wall-Truth|.  Assumes NSCBC/Wall already run; (re)runs Tall.
cd "$(dirname "$0")/../../.." || exit 1
FX=ext/AMReX-Codes/amrex/Tools/Plotfile/fextract.3d.ex
./configure --dim=3 --comp=g++ >/dev/null 2>&1
make -j12 bin/hydro2-3d-g++ >/tmp/b3.log 2>&1
echo "=== run Tall (free-space truth) ==="
mpirun -np 6 ./bin/hydro2-3d-g++ tests/FlowAcoustic/input_AcousticPulse3D_Tall >/tmp/ap_tall.log 2>&1
grep -iE 'finalized|ABORT|nan' /tmp/ap_tall.log | tail -1

last () { ls -d "$1"/[0-9]*cell 2>/dev/null | sort | tail -1; }
$FX -d 2 -x 0.0 -y 0.0 -v pressure -s /tmp/pz_n.txt "$(last tests/FlowAcoustic/output_Pulse3D_NSCBC)" 2>/dev/null
$FX -d 2 -x 0.0 -y 0.0 -v pressure -s /tmp/pz_w.txt "$(last tests/FlowAcoustic/output_Pulse3D_Wall)"  2>/dev/null
$FX -d 2 -x 0.0 -y 0.0 -v pressure -s /tmp/pz_t.txt "$(last tests/FlowAcoustic/output_Pulse3D_Tall)"  2>/dev/null

# Truth restricted to |z|<=0.5, keyed by z (rounded) -> pressure.
awk 'BEGIN{} !/^#/ && NF>=2 { z=$1+0; if (z>=-0.5 && z<=0.5){ key=sprintf("%.4f",z); t[key]=$2 } } END{ for(k in t) print k, t[k] }' /tmp/pz_t.txt > /tmp/truth.txt

cmp_to_truth () { # $1 = slice file
  awk -v tf=/tmp/truth.txt 'BEGIN{ while((getline line < tf)>0){ split(line,a," "); T[a[1]]=a[2] } }
    !/^#/ && NF>=2 { key=sprintf("%.4f",$1+0); if (key in T){ d=$2-T[key]; if(d<0)d=-d; if(d>m)m=d } } END{ printf "%.5g", m }' "$1"
}
echo "  max|NSCBC - Truth| over z in [-0.5,0.5] = $(cmp_to_truth /tmp/pz_n.txt)"
echo "  max|Wall  - Truth| over z in [-0.5,0.5] = $(cmp_to_truth /tmp/pz_w.txt)"
