#!/bin/bash
# =============================================================================
# Generate the 3D Marmottant static-Laplace radius sweep (octant).
#
# Expands TEMPLATE_input_3D_Marmottant once per radius into input_R<nm>, with
# plot_file pointing at INCLINE by default.  Same radii, fluids and shell as
# the 2D sweep (tests/FlowMarmottant/TEMPLATES/RUN_MARMOTTANT_SWEEP.bash), so
# the 2D and 3D results compare case-for-case.
#
#   ./GENERATE_3D_INPUTS.bash                      # INCLINE paths, 20 ms
#   OUTROOT=$PWD/../../../bin/tests/FlowMarmottant/3D_Bubble ./GENERATE_3D_INPUTS.bash   # local
#   STOP=5e-3 PLOT_DT=2.5e-4 ./GENERATE_3D_INPUTS.bash
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$HERE/TEMPLATE_input_3D_Marmottant"

RATIOS=${RATIOS:-"0.85 0.93 1.00 1.012 1.025 1.038 1.051 1.064 1.15 1.35"}
OUTROOT=${OUTROOT:-/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/3D_Bubble}
DEST=${DEST:-$HERE}

RBUCK=${RBUCK:-1.0e-3}
RHO_L=1000.0; RHO_G=1.0
GAM_L=2.35;   PINF_L=1.0e9;   GAM_G=1.4
CP_L=4186.0;  CV_L=1781.0;    CP_G=1000.0;  CV_G=714.286
MU_L=1.0e-3;  MU_G=1.8e-5
PLIQ=101325.0
CHI=${CHI:-0.55}; SBREAK=${SBREAK:-0.073}
MARMOTTANT=${MARMOTTANT:-1}; CAP_CLOSURE=${CAP_CLOSURE:-1}; LIMITER=${LIMITER:-vanleer}
MAXLEV=${MAXLEV:-2}; NCELL=${NCELL:-32}; EPSDX=${EPSDX:-4}; BOXR=${BOXR:-2.5}
STOP=${STOP:-2.0e-2}
PLOT_DT=${PLOT_DT:-1.0e-3}          # 21 plotfiles: 3D plotfiles are large
THERMO_INT=${THERMO_INT:-50}        # coarse steps between thermo.dat rows
DT_INIT=${DT_INIT:-1.0e-12}; DT_MIN=${DT_MIN:-1.0e-16}

[ -f "$TEMPLATE" ] || { echo "missing $TEMPLATE"; exit 1; }

read -r L DXFINE EPSILON RRUPT LREF <<<"$(python3 -c "
import math
Rb=$RBUCK; L=$BOXR*Rb
dx=L/($NCELL*2**$MAXLEV); eps=$EPSDX*dx
rr=Rb*math.sqrt(1+$SBREAK/$CHI)
lref=(2.0*L)/20.0        # full physical width / 20 (2D-validated value)
print('%.8e %.8e %.8e %.8e %.6e'%(L,dx,eps,rr,lref))
")"

echo "3D Marmottant sweep: octant [0,$L]^3  dx_fine=$DXFINE  eps=$EPSILON  L_ref=$LREF"
echo "  stop=$STOP  plot_dt=$PLOT_DT  thermo every $THERMO_INT steps"
echo "  plot_file root: $OUTROOT"
n=0
for RATIO in $RATIOS; do
  read -r R0 SIG DP PGAS REG NAME R0DX <<<"$(python3 -c "
Rb=$RBUCK; chi=$CHI; sb=$SBREAK; R=Rb*$RATIO
s=chi*((R/Rb)**2-1.0); s=sb if s>=sb else max(s,0.0)
dp=2.0*s/R                                   # 3D: two principal curvatures
reg='buckled' if s<=0 else ('ruptured' if s>=sb else 'elastic')
print('%.8e %.8e %.6f %.6f %s R%09d %.1f'%(R,s,dp,$PLIQ+dp,reg,round(R*1e9),R/$DXFINE))
")"
  OUT="$DEST/input_${NAME}"
  sed -e "s|@NAMETAG@|marm3d_${NAME,,}|g" -e "s|@OUTPUT_PATH@|$OUTROOT/output_$NAME|g" \
      -e "s|@R0@|$R0|g" -e "s|@RBUCK@|$RBUCK|g" -e "s|@RRUPT@|$RRUPT|g" -e "s|@RRATIO@|$RATIO|g" \
      -e "s|@CHI@|$CHI|g" -e "s|@SBREAK@|$SBREAK|g" -e "s|@SIGMA0@|$SIG|g" -e "s|@REGIME@|$REG|g" \
      -e "s|@DP@|$DP|g" -e "s|@PGAS@|$PGAS|g" -e "s|@PLIQ@|$PLIQ|g" \
      -e "s|@EPSILON@|$EPSILON|g" -e "s|@EPSDX@|$EPSDX|g" -e "s|@DXFINE@|$DXFINE|g" -e "s|@R0DX@|$R0DX|g" \
      -e "s|@L@|$L|g" -e "s|@LREF@|$LREF|g" -e "s|@NCELL@|$NCELL|g" -e "s|@MAXLEV@|$MAXLEV|g" \
      -e "s|@STOP_TIME@|$STOP|g" -e "s|@PLOT_DT@|$PLOT_DT|g" -e "s|@THERMO_INT@|$THERMO_INT|g" \
      -e "s|@DT_INIT@|$DT_INIT|g" -e "s|@DT_MIN@|$DT_MIN|g" \
      -e "s|@RHO_L@|$RHO_L|g" -e "s|@RHO_G@|$RHO_G|g" -e "s|@GAM_L@|$GAM_L|g" -e "s|@PINF_L@|$PINF_L|g" \
      -e "s|@GAM_G@|$GAM_G|g" -e "s|@CP_L@|$CP_L|g" -e "s|@CV_L@|$CV_L|g" -e "s|@CP_G@|$CP_G|g" -e "s|@CV_G@|$CV_G|g" \
      -e "s|@MU_L@|$MU_L|g" -e "s|@MU_G@|$MU_G|g" -e "s|@MARMOTTANT@|$MARMOTTANT|g" \
      -e "s|@CAP_CLOSURE@|$CAP_CLOSURE|g" -e "s|@LIMITER@|$LIMITER|g" \
      "$TEMPLATE" > "$OUT"
  left=$(grep -c '@[A-Z0-9_]*@' "$OUT")
  [ "$left" -ne 0 ] && { echo "FATAL: $left unfilled tokens in $OUT"; grep -o '@[A-Z0-9_]*@' "$OUT" | sort -u; exit 1; }
  printf "  %-11s R/Rb=%-6s %-9s sigma=%.6f  dp=2s/R=%9.4f Pa  R0/dx=%s\n" "$NAME" "$RATIO" "$REG" "$SIG" "$DP" "$R0DX"
  n=$((n+1))
done
echo "wrote $n inputs to $DEST"
