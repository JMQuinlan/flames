#!/bin/bash
#
# Run the full FlowLaplace surface-tension matrix (R x sigma) back-to-back from
# a single template.  Resolves paths relative to its own location, so it can be
# invoked from anywhere:
#
#   bash tests/FlowLaplace/TEMPLATES/RUN_LAPLACE.bash
#   NP=4 bash tests/FlowLaplace/TEMPLATES/RUN_LAPLACE.bash
#   DRY_RUN=1 bash tests/FlowLaplace/TEMPLATES/RUN_LAPLACE.bash   # expand inputs, don't run
#   KEEP=1 bash tests/FlowLaplace/TEMPLATES/RUN_LAPLACE.bash      # keep the expanded inputs
#
# For each (R, sigma) the derived quantities (domain, epsilon, capillary
# timescale, stop_time, dt caps, gas pressure) are computed here by awk using
# the SAME formulas as tests/FlowLaplace/reference/gen_inputs.py, substituted
# into TEMPLATES/input_Laplace, and run.  Output lands in
#   bin/tests/FlowLaplace/output_R<R>_Sigma<sigma>
# which the analysis scripts already search.  Afterwards:
#   cd tests/FlowLaplace/reference && python analyze_Laplace_AGG.py

set -e

# -----------------------------
# INPUTS  (edit / comment to subset the matrix)
# -----------------------------
R_VALUES=(
  1.0
  0.001
  0.000001
)
SIGMA_VALUES=(
  0.0
  1.0
  10.0
)

NP=${NP:-6}
EXEC=${EXEC:-./hydro2-2d-g++}

# Fluid constants -- MUST match gen_inputs.py / input_Laplace.
RHO_LIQ=0.991
P_LIQUID=1.0
EOS0_GAMMA=5.5
EOS0_P0=1.505

# -----------------------------
# PATHS
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/input_Laplace"
BIN_DIR="$(cd "$SCRIPT_DIR/../../../bin" && pwd)"

if [ ! -f "$TEMPLATE" ]; then
  echo "ERROR: template not found: $TEMPLATE"
  exit 1
fi
cd "$BIN_DIR"
if [ -z "$DRY_RUN" ] && [ ! -x "$EXEC" ]; then
  echo "ERROR: executable not found or not runnable: $(pwd)/$EXEC"
  echo "       (use DRY_RUN=1 to only expand the input files)"
  exit 1
fi

# -----------------------------
# MAIN LOOP
# -----------------------------
overall_start=$SECONDS

for R in "${R_VALUES[@]}"; do
  for SIGMA in "${SIGMA_VALUES[@]}"; do

    # Compute every derived quantity for this (R, sigma) in one awk pass.
    # Mirrors gen_inputs.py:  domain half-width 2R, 64+1 AMR levels -> dx=R/32,
    # epsilon = 2*dx, capillary (or acoustic) tau, stop_time = 15 tau, etc.
    read RNAME SNAME EPSILON HALF NEG PGAS STOP PLOTDT DTI DTMAX DTMIN APPLYST DP TAU DXFINE BOX < <(
      awk -v R="$R" -v S="$SIGMA" \
          -v rho="$RHO_LIQ" -v pliq="$P_LIQUID" -v g0="$EOS0_GAMMA" -v p0="$EOS0_P0" '
        function floor(x){ return (x==int(x)) ? x : (x<0 ? int(x)-1 : int(x)) }
        function namesci(v,   e,m){
          if (v==0) return "0.0"
          e = floor(log(v)/log(10))
          m = v / (10 ^ e)
          if (m >= 9.99995) { m = m/10; e = e+1 }   # floor undershot a decade
          return sprintf("%.1fe%d", m, e)
        }
        BEGIN{
          CONVFMT="%.10g"; OFMT="%.10g"
          cliq   = sqrt(g0*(pliq+p0)/rho)
          half   = 2*R; neg = -half; box = 4*R
          dxfine = 4*R/128; eps = 2*dxfine
          dp     = S/R; pgas = pliq + dp
          if (S > 0) tau = sqrt(rho*R*R*R/S); else tau = R/cliq
          stop   = 15*tau; plotdt = stop/50
          dti    = stop/1000; dtmax = stop/200; dtmin = stop/1e8
          applyst = (S > 0) ? 1 : 0
          printf "%s %s %s %s %s %s %s %s %s %s %s %d %s %s %s %s\n", \
            namesci(R), namesci(S), eps, half, neg, pgas, stop, plotdt, \
            dti, dtmax, dtmin, applyst, dp, tau, dxfine, box
        }'
    )

    NAME="R${RNAME}_Sigma${SNAME}"
    OUTPUT_PATH="./tests/FlowLaplace/output_${NAME}"
    INPUT_FILE="tmp_Laplace_${NAME}.in"

    sed -e "s|@NAMETAG@|${NAME,,}|g" \
        -e "s|@OUTPUT_PATH@|${OUTPUT_PATH}|g" \
        -e "s|@R0@|${R}|g" \
        -e "s|@SIGMA@|${SIGMA}|g" \
        -e "s|@EPSILON@|${EPSILON}|g" \
        -e "s|@NEG_HALF@|${NEG}|g" \
        -e "s|@HALF@|${HALF}|g" \
        -e "s|@PGAS@|${PGAS}|g" \
        -e "s|@STOP_TIME@|${STOP}|g" \
        -e "s|@PLOT_DT@|${PLOTDT}|g" \
        -e "s|@DT_INIT@|${DTI}|g" \
        -e "s|@DT_MAX@|${DTMAX}|g" \
        -e "s|@DT_MIN@|${DTMIN}|g" \
        -e "s|@APPLY_ST@|${APPLYST}|g" \
        -e "s|@DP@|${DP}|g" \
        -e "s|@TAU@|${TAU}|g" \
        -e "s|@DXFINE@|${DXFINE}|g" \
        -e "s|@BOX@|${BOX}|g" \
        "$TEMPLATE" > "$INPUT_FILE"

    echo
    echo "============================================================"
    echo "=== Laplace  ${NAME}   (R=${R}, sigma=${SIGMA}, dp=${DP}, NP=${NP})"
    echo "============================================================"

    if [ -n "$DRY_RUN" ]; then
      echo "    [dry-run] wrote ${BIN_DIR}/${INPUT_FILE} (not running)"
      continue
    fi

    case_start=$SECONDS
    mpirun -np "$NP" "$EXEC" "$INPUT_FILE" || true
    echo "--- ${NAME} elapsed: $((SECONDS - case_start)) s"

    if [ -z "$KEEP" ]; then
      rm -f "$INPUT_FILE"
    fi
  done
done

echo
echo "=== ALL LAPLACE RUNS COMPLETE  (total: $((SECONDS - overall_start)) s) ==="
echo "    analyze with: cd tests/FlowLaplace/reference && python analyze_Laplace_AGG.py"
