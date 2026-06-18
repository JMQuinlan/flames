#!/bin/bash
# Decisive 3D check: does 3D Garrick_x (and _z) match the TRUSTED 2D solver at the
# same time?  Runs 2D Garrick to t=0.01, extracts the density profile along the
# Riemann axis for 2D / 3D-x / 3D-z, and reports max|3D - 2D| over the profile.
cd "$(dirname "$0")/../../.." || exit 1
FX2=ext/AMReX-Codes/amrex/Tools/Plotfile/fextract.2d.ex
FX3=ext/AMReX-Codes/amrex/Tools/Plotfile/fextract.3d.ex

echo "=== run 2D Garrick to t=0.01 ==="
./configure --dim=2 --comp=g++ >/dev/null 2>&1
make -j12 bin/hydro2-2d-g++ >/tmp/b.log 2>&1
mpirun -np 6 ./bin/hydro2-2d-g++ tests/FlowRiemannUnitTests/input_Garrick \
       plot_file=./tests/FlowRiemannUnitTests/output_Garrick_t01 stop_time=0.01 >/tmp/g2.log 2>&1
grep -iE 'finalized|ABORT' /tmp/g2.log | tail -1

# density column = 12 in these plotfiles (eta,rho_eta0,rho_eta1,etadot,energy0,
# energy1,pressure,velocityx,velocityy,vorticity,density,...) -> col 12 (x is col 1).
$FX2 -d 0 -v density -s /tmp/p2.txt tests/FlowRiemannUnitTests/output_Garrick_t01/00100cell 2>/dev/null
$FX3 -d 0 -v density -s /tmp/px.txt tests/FlowRiemannUnitTests/output_Garrick_x/00100cell 2>/dev/null
$FX3 -d 2 -x 0.0 -y 0.0 -v density -s /tmp/pz.txt tests/FlowRiemannUnitTests/output_Garrick_z/00100cell 2>/dev/null

col () { grep -vE '^#|^$' "$1" | awk '{print $2}'; }
echo "=== max |3D - 2D| density over the 800-pt Riemann profile ==="
paste <(col /tmp/px.txt) <(col /tmp/p2.txt) | awk '{d=$1-$2;if(d<0)d=-d;if(d>m){m=d;mi=NR}} END{printf "  Garrick_x vs 2D : max=%.4g at row %d\n",m,mi}'
paste <(col /tmp/pz.txt) <(col /tmp/p2.txt) | awk '{d=$1-$2;if(d<0)d=-d;if(d>m){m=d;mi=NR}} END{printf "  Garrick_z vs 2D : max=%.4g at row %d\n",m,mi}'
echo "=== interface-region rows (398-402): 2D  |  3D-x  |  3D-z ==="
paste <(col /tmp/p2.txt) <(col /tmp/px.txt) <(col /tmp/pz.txt) | sed -n '398,402p'
