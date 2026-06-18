#!/bin/bash
# 3D Garrick diagnostics: dump IC + evolved profiles and transverse uniformity.
# fextract writes its slice correctly but segfaults on exit -- do NOT use set -e.
# MUST use the 3D-compiled tool on 3D plotfiles -- a 2D-compiled fextract
# silently misreads 3D files (shifted columns, empty dir-2 slices).
FX=ext/AMReX-Codes/amrex/Tools/Plotfile/fextract.3d.ex
GXIC=tests/FlowRiemannUnitTests/output_Garrick_x/00000cell
GX=${1:-tests/FlowRiemannUnitTests/output_Garrick_x/00100cell}
GZ=${2:-tests/FlowRiemannUnitTests/output_Garrick_z/00100cell}

dump3 () { grep -vE '^#|^$' "$1" | sed -n "$2"; }

echo "### Garrick_x IC (00000cell) density,eta along x  [expect rho~1.241 left, ~0.991 right] ###"
$FX -d 0 -v "density eta" -s /tmp/ic.txt "$GXIC" 2>/dev/null
echo "  left  : $(dump3 /tmp/ic.txt '1p')"
echo "  center: $(dump3 /tmp/ic.txt '400p')"
echo "  right : $(dump3 /tmp/ic.txt '800p')"

echo "### Garrick_x evolved (t=0.01) density along x ###"
$FX -d 0 -v "density" -s /tmp/gx_x.txt "$GX" 2>/dev/null
echo "  left  : $(dump3 /tmp/gx_x.txt '1p')"
echo "  center: $(dump3 /tmp/gx_x.txt '400p')"
echo "  right : $(dump3 /tmp/gx_x.txt '800p')"

echo "### Garrick_x transverse uniformity: density along y (dir 1) ###"
$FX -d 1 -v density -s /tmp/gx_y.txt "$GX" 2>/dev/null
grep -vE '^#|^$' /tmp/gx_y.txt | awk 'NR==1{mn=mx=$2} {if($2<mn)mn=$2; if($2>mx)mx=$2} END{printf "  min=%.10g max=%.10g spread=%.3g\n",mn,mx,mx-mn}'

echo "### Garrick_z evolved (t=0.01) density along z (dir 2, through x=0 y=0) ###"
$FX -d 2 -x 0.0 -y 0.0 -v "density" -s /tmp/gz_z.txt "$GZ" 2>/dev/null
echo "  rows extracted: $(grep -vcE '^#|^$' /tmp/gz_z.txt)"
echo "  left  : $(dump3 /tmp/gz_z.txt '1p')"
echo "  center: $(dump3 /tmp/gz_z.txt '400p')"
echo "  right : $(dump3 /tmp/gz_z.txt '800p')"
