#!/bin/bash
# =============================================================================
# Marmottant coated-bubble static Laplace sweep.
#
# Expands TEMPLATES/input_Marmottant_Sweep once per radius and runs them, N at a
# time.  Radii are given as RATIOS of R_buckling, so the regime map (buckled /
# elastic / ruptured) is identical whatever physical scale you pick.
#
#   ./RUN_MARMOTTANT_SWEEP.bash                  # physical air/water, R_buck=1mm
#   UNITS=code ./RUN_MARMOTTANT_SWEEP.bash       # the original scaled-unit set
#   RBUCK=2.0e-6 STOP=1.0e-6 ./RUN_...           # real UCA microbubble
#   NP=6 NPAR=5 ./RUN_...                        # 5 cases at a time, 6 ranks each
#   DRYRUN=1 ./RUN_...                           # write inputs, run nothing
#
# Every case starts in Laplace equilibrium, so the correct answer is that
# NOTHING MOVES.  p_gas is derived per radius from sigma_eff(R0)/R0.
# =============================================================================
set -u

# ----------------------------- what to sweep --------------------------------
# R0 / R_buckling.  Default: 3 buckled, 5 across the elastic ramp, 2 ruptured.
RATIOS=${RATIOS:-"0.85 0.93 1.00 1.012 1.025 1.038 1.051 1.064 1.15 1.35"}

UNITS=${UNITS:-physical}          # physical | code
RBUCK=${RBUCK:-}                  # R_buckling [m]; default set per UNITS
MARMOTTANT=${MARMOTTANT:-1}       # 0 = constant sigma control
CAP_CLOSURE=${CAP_CLOSURE:-1}     # 1 = Schmidmayer split closure
LIMITER=${LIMITER:-vanleer}
BCMODE=${BCMODE:-nscbc}        # nscbc | neumann | dirichlet
MAXLEV=${MAXLEV:-2}
EPSDX=${EPSDX:-4}                 # diffuse half-width in finest cells
BOXR=${BOXR:-2.5}                 # half-domain = BOXR * R_buckling
NP=${NP:-6}                       # MPI ranks per case
NPAR=${NPAR:-5}                   # cases in flight
EXEC=${EXEC:-./hydro2-2d-g++}
TAG=${TAG:-marmsweep}
DRYRUN=${DRYRUN:-0}

# --------------------------- material properties ----------------------------
if [ "$UNITS" = "code" ]; then
    RBUCK=${RBUCK:-0.018}
    RHO_L=10.0;   RHO_G=1.0
    GAM_L=7.15;   PINF_L=5000.0;  GAM_G=1.4
    CP_L=1000.0;  CV_L=714.286;   CP_G=1000.0;  CV_G=714.286
    MU_L=0.15;    MU_G=0.015
    PLIQ=500.0
    CHI=14.56;    SBREAK=7.28
    STOP=${STOP:-1.5e-2};  DT_INIT=${DT_INIT:-1.0e-6};  DT_MIN=${DT_MIN:-1.0e-12}
else
    # air / water at STP.  Water EOS is the set FlowLaplace and the RPE
    # collapse test use -- gamma=2.35, p_inf=1e9 -- NOT gamma=4.4/6e8.
    RBUCK=${RBUCK:-1.0e-3}
    RHO_L=1000.0; RHO_G=1.0
    GAM_L=2.35;   PINF_L=1.0e9;   GAM_G=1.4
    CP_L=4186.0;  CV_L=1781.0;    CP_G=1000.0;  CV_G=714.286
    MU_L=1.0e-3;  MU_G=1.8e-5
    PLIQ=101325.0
    CHI=${CHI:-0.55};  SBREAK=${SBREAK:-0.073}     # Marmottant 2005
    STOP=${STOP:-1.0e-2};  DT_INIT=${DT_INIT:-1.0e-12};  DT_MIN=${DT_MIN:-1.0e-16}
fi
PLOT_DT=${PLOT_DT:-$(python3 -c "print('%.6e'%($STOP/20))")}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/input_Marmottant_Sweep"
TEST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$(cd "$SCRIPT_DIR/../../../bin" && pwd)"
GEN_DIR="$TEST_DIR/GENERATED_$TAG"
mkdir -p "$GEN_DIR"
[ -f "$TEMPLATE" ] || { echo "missing template: $TEMPLATE"; exit 1; }

# ---------------------------- derived geometry ------------------------------
read -r HALF NEG DXFINE EPSILON RRUPT TCAP LREF <<<"$(python3 -c "
import math
Rb=$RBUCK; box=$BOXR; lev=$MAXLEV; epsdx=$EPSDX
half=box*Rb; dx=2*half/(64*2**lev); eps=epsdx*dx
rr=Rb*math.sqrt(1+$SBREAK/$CHI)
tcap=math.sqrt($RHO_L*Rb**3/max($SBREAK,1e-30))
print('%.8e %.8e %.8e %.8e %.8e %.6e %.6e'%(half,-half,dx,eps,rr,tcap,2*half))
")"

echo "=============================================================="
echo " Marmottant sweep   UNITS=$UNITS   R_buckling=$RBUCK m"
echo "   R_rupture = $RRUPT m      chi=$CHI  sigma_break=$SBREAK"
echo "   domain +-$HALF   dx_fine=$DXFINE   eps=$EPSILON (eps/dx=$EPSDX)"
echo "   capillary time = $TCAP s   stop_time=$STOP  ($NP ranks x $NPAR cases)"
echo "   inputs -> $GEN_DIR"
echo "=============================================================="

# ---- boundary-condition block -------------------------------------------
# nscbc  : characteristic outflow, holds far-field pressure (conservative)
# neumann: zero-gradient.  Runs, but is NOT conservative -- measured to leak
#          liquid mass into the domain (+0.42%/8.5ms), pressurising a closed box
#          and compressing every bubble uniformly regardless of sigma.
if [ "$BCMODE" = "dirichlet" ]; then
  # Full domain, walls held at ambient pressure with zero momentum.
  # Primitive BC path (bc.primitive=1): pressure is imposed directly, so the
  # walls are a fixed-pressure reservoir; momentum Dirichlet 0 makes them
  # non-permeable.  No characteristic decomposition, no NSCBC feedback path,
  # and no opposing-outflow pair -- the configuration the centreline wave
  # pattern was traced to.  Density/energy stay zero-gradient.
  BCBLOCK=$(cat <<BCEOF
bc.primitive = 1

pressure.bc.constant.p_inf = $PLIQ
pressure.bc.type.xlo = dirichlet
pressure.bc.val.xlo  = "p_inf"
pressure.bc.type.xhi = dirichlet
pressure.bc.val.xhi  = "p_inf"
pressure.bc.type.ylo = dirichlet
pressure.bc.val.ylo  = "p_inf"
pressure.bc.type.yhi = dirichlet
pressure.bc.val.yhi  = "p_inf"

momentum.bc.type.xlo = dirichlet dirichlet
momentum.bc.val.xlo  = "0.0" "0.0"
momentum.bc.type.xhi = dirichlet dirichlet
momentum.bc.val.xhi  = "0.0" "0.0"
momentum.bc.type.ylo = dirichlet dirichlet
momentum.bc.val.ylo  = "0.0" "0.0"
momentum.bc.type.yhi = dirichlet dirichlet
momentum.bc.val.yhi  = "0.0" "0.0"

density.bc.type.xlo = neumann
density.bc.type.xhi = neumann
density.bc.type.ylo = neumann
density.bc.type.yhi = neumann
energy.bc.type.xlo  = neumann
energy.bc.type.xhi  = neumann
energy.bc.type.ylo  = neumann
energy.bc.type.yhi  = neumann
eta.bc.type.xlo = neumann
eta.bc.type.xhi = neumann
eta.bc.type.ylo = neumann
eta.bc.type.yhi = neumann
BCEOF
)
elif [ "$BCMODE" = "neumann" ]; then
  BCBLOCK=$(cat <<'BCEOF'
density.bc.type.xhi  = neumann
density.bc.type.xlo  = neumann
density.bc.type.ylo  = neumann
density.bc.type.yhi  = neumann
energy.bc.type.xhi   = neumann
energy.bc.type.xlo   = neumann
energy.bc.type.ylo   = neumann
energy.bc.type.yhi   = neumann
momentum.bc.type.xhi = neumann neumann
momentum.bc.type.xlo = neumann neumann
momentum.bc.type.ylo = neumann neumann
momentum.bc.type.yhi = neumann neumann
BCEOF
)
else
  BCBLOCK=$(cat <<BCEOF
density.bc.type.xhi  = nscbc_outflow
density.bc.type.xlo  = nscbc_outflow
density.bc.type.ylo  = nscbc_outflow
density.bc.type.yhi  = nscbc_outflow
energy.bc.type.xhi   = nscbc_outflow
energy.bc.type.xlo   = nscbc_outflow
energy.bc.type.ylo   = nscbc_outflow
energy.bc.type.yhi   = nscbc_outflow
momentum.bc.type.xhi = nscbc_outflow nscbc_outflow
momentum.bc.type.xlo = nscbc_outflow nscbc_outflow
momentum.bc.type.ylo = nscbc_outflow nscbc_outflow
momentum.bc.type.yhi = nscbc_outflow nscbc_outflow

nscbc.xlo.type = outflow
nscbc.xlo.target_p = $PLIQ
nscbc.xlo.sigma = 0.15
nscbc.xlo.beta  = 0.5
nscbc.xlo.L_ref = $LREF
nscbc.xlo.target_u = 0.0
nscbc.xlo.target_v = 0.0
nscbc.xhi.type = outflow
nscbc.xhi.target_p = $PLIQ
nscbc.xhi.sigma = 0.15
nscbc.xhi.beta  = 0.5
nscbc.xhi.L_ref = $LREF
nscbc.xhi.target_u = 0.0
nscbc.xhi.target_v = 0.0
nscbc.ylo.type = outflow
nscbc.ylo.target_p = $PLIQ
nscbc.ylo.sigma = 0.15
nscbc.ylo.beta  = 0.5
nscbc.ylo.L_ref = $LREF
nscbc.ylo.target_u = 0.0
nscbc.ylo.target_v = 0.0
nscbc.yhi.type = outflow
nscbc.yhi.target_p = $PLIQ
nscbc.yhi.sigma = 0.15
nscbc.yhi.beta  = 0.5
nscbc.yhi.L_ref = $LREF
nscbc.yhi.target_u = 0.0
nscbc.yhi.target_v = 0.0
nscbc.small = 1.0e-10
BCEOF
)
fi
export BCBLOCK BCMODE

JOBS="$GEN_DIR/.jobs"; : > "$JOBS"
for RATIO in $RATIOS; do
    read -r R0 SIGMA0 DP PGAS REGIME NAME <<<"$(python3 -c "
Rb=$RBUCK; chi=$CHI; sb=$SBREAK; ratio=$RATIO
R=Rb*ratio
s=chi*((R/Rb)**2-1.0)
s=sb if s>=sb else max(s,0.0)
dp=s/R
reg='buckled' if s<=0 else ('ruptured' if s>=sb else 'elastic')
print('%.8e %.8e %.6f %.6f %s R%09d'%(R,s,dp,$PLIQ+dp,reg,round(R*1e9)))
")"
    OUT="$GEN_DIR/input_${NAME}"
    sed -e "s|@NAMETAG@|${NAME,,}|g"        -e "s|@OUTPUT_PATH@|$BIN_DIR/tests/FlowMarmottant/$TAG/$NAME|g" \
        -e "s|@R0@|$R0|g"                   -e "s|@RBUCK@|$RBUCK|g" \
        -e "s|@RRUPT@|$RRUPT|g"             -e "s|@RRATIO@|$RATIO|g" \
        -e "s|@CHI@|$CHI|g"                 -e "s|@SBREAK@|$SBREAK|g" \
        -e "s|@SIGMA0@|$SIGMA0|g"           -e "s|@REGIME@|$REGIME|g" \
        -e "s|@DP@|$DP|g"                   -e "s|@PGAS@|$PGAS|g" \
        -e "s|@PLIQ@|$PLIQ|g"               -e "s|@EPSILON@|$EPSILON|g" \
        -e "s|@EPSDX@|$EPSDX|g"             -e "s|@DXFINE@|$DXFINE|g" \
        -e "s|@HALF@|$HALF|g"               -e "s|@NEG_HALF@|$NEG|g" \
        -e "s|@BOX@|$LREF|g"                -e "s|@LREF@|$LREF|g" \
        -e "s|@MAXLEV@|$MAXLEV|g"           -e "s|@TCAP@|$TCAP|g" \
        -e "s|@STOP_TIME@|$STOP|g"          -e "s|@PLOT_DT@|$PLOT_DT|g" \
        -e "s|@DT_INIT@|$DT_INIT|g"         -e "s|@DT_MIN@|$DT_MIN|g" \
        -e "s|@RHO_L@|$RHO_L|g"             -e "s|@RHO_G@|$RHO_G|g" \
        -e "s|@GAM_L@|$GAM_L|g"             -e "s|@PINF_L@|$PINF_L|g" \
        -e "s|@GAM_G@|$GAM_G|g"             -e "s|@CP_L@|$CP_L|g" \
        -e "s|@CV_L@|$CV_L|g"               -e "s|@CP_G@|$CP_G|g" \
        -e "s|@CV_G@|$CV_G|g"               -e "s|@MU_L@|$MU_L|g" \
        -e "s|@MU_G@|$MU_G|g"               -e "s|@MARMOTTANT@|$MARMOTTANT|g" \
        -e "s|@CAP_CLOSURE@|$CAP_CLOSURE|g" -e "s|@LIMITER@|$LIMITER|g" \
        -e "s|@BCMODE@|$BCMODE|g" \
        "$TEMPLATE" > "$OUT.tmp"
    printf '%s\n' "$BCBLOCK" > "$GEN_DIR/.bcblock"
    awk '/@BCBLOCK@/{while((getline l < "'"$GEN_DIR"'/.bcblock")>0) print l; next} {print}' "$OUT.tmp" > "$OUT"
    rm -f "$OUT.tmp"
    if grep -q '@[A-Z_]*@' "$OUT"; then
        echo "  ERROR unfilled tokens in $OUT:"; grep -o '@[A-Z_]*@' "$OUT" | sort -u | sed 's/^/    /'; exit 1
    fi
    printf '%s\n' "$OUT" >> "$JOBS"
    printf '  %-12s ratio=%-6s R0=%-13s sigma=%-11s dp=%-10s %s\n' "$NAME" "$RATIO" "$R0" "$SIGMA0" "$DP" "$REGIME"
done

NCASE=$(wc -l < "$JOBS")
echo "--------------------------------------------------------------"
echo " wrote $NCASE inputs"
[ "$DRYRUN" = "1" ] && { echo " DRYRUN=1, not running."; exit 0; }

run_one() {
    set -u
    INP="$1"
    NAME=$(basename "$INP" | sed 's/^input_//')
    LOG="$(dirname "$INP")/${NAME}.log"
    mpirun -np ${NP} --oversubscribe ${EXEC} "$INP" > "$LOG" 2>&1
    rc=$?
    ab=$(grep -ac MPI_ABORT "$LOG" 2>/dev/null)
    t=$(grep -av btl_tcp "$LOG" | grep -aoE "TIME = [0-9.e-]+" | tail -1 | awk '{print $3}')
    echo "  $NAME rc=$rc aborts=${ab:-0} last_t=${t:-none}"
}
export -f run_one
export NP EXEC

cd "$BIN_DIR" || exit 1
echo " running $NCASE cases, $NPAR at a time, $NP ranks each"
xargs -P "$NPAR" -n 1 bash -c 'run_one "$@"' _ < "$JOBS"
echo " SWEEP_COMPLETE  ->  $BIN_DIR/tests/FlowMarmottant/$TAG"
