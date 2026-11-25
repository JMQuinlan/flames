#!/bin/bash
#
# FlowWedge Mach sweep from a single template (same pattern as
# tests/FlowLaplace/TEMPLATES/RUN_LAPLACE.bash).  For every Mach number in
# MACH_VALUES the Mach-dependent quantities are computed here by awk,
# substituted into TEMPLATES/input_Wedge, and run.  Resolves paths relative to
# its own location, so it can be invoked from anywhere:
#
#   bash tests/FlowWedge/TEMPLATES/sweep_mach.bash
#   NP=4 bash tests/FlowWedge/TEMPLATES/sweep_mach.bash
#   MACH="2.0 20.0" bash tests/FlowWedge/TEMPLATES/sweep_mach.bash    # override the list
#   DRY_RUN=1 bash tests/FlowWedge/TEMPLATES/sweep_mach.bash         # expand inputs, don't run
#   KEEP=1 bash tests/FlowWedge/TEMPLATES/sweep_mach.bash            # keep the expanded inputs
#   OUTPUT_ROOT=/mmfs1/home/ttryon/flames/bin/tests/FlowWedge \
#   LAUNCH="srun -n 128 --mpi=pmi2" bash tests/FlowWedge/TEMPLATES/sweep_mach.bash   # INCLINE
#
# Output lands in  $OUTPUT_ROOT/output_Ma<M>  (default ./tests/FlowWedge under bin/),
# the naming the analysis already searches.  Afterwards:
#   python3 tests/FlowWedge/reference/analyze_mach_sweep.py ["<OUTPUT_ROOT>/output_Ma{ma}"]
#
# Per-Mach rules (freestream rho = 1, p = 1/gamma so c = 1 and U = Mach;
# domain length L = 6, so one flow-through is L/U = 6/M):
#   stop_time  = 5 L/U (= 30/M) for M >= 3, 6 L/U (= 36/M) below M = 3.
#                (beta steady to < 0.01 deg per time unit by ~4 L/U at M 3 and 5;
#                 M 2 still creeps at 5 L/U, so the low-Mach cases get 6)
#   plot_dt    = min(0.5, stop_time/10)   (>= ~10 plotfiles for the steadiness check)
#   dt_max     = 3e-3/M,  brinkman = 400*M  (as the original FlowWedge decks)
#   energy     = 1.785714 + 0.5 M^2       (header comment only)

set -e

# -----------------------------
# INPUTS  (edit / comment to choose the sweep)
# -----------------------------
MACH_VALUES=(
  1.2
  2.0
  3.0
  5.0
  # 1.5 2.5 3.5 4.0 4.5 6.0 7.0 8.0 10.0 15.0 20.0 25.0 30.0 35.0
)
[ -n "$MACH" ] && read -r -a MACH_VALUES <<< "$MACH"

NP=${NP:-2}
EXEC=${EXEC:-./hydro2-2d-g++}
LAUNCH=${LAUNCH:-mpirun -np $NP}
OUTPUT_ROOT=${OUTPUT_ROOT:-./tests/FlowWedge}
INPUT_DIR=${INPUT_DIR:-.}           # where the expanded tmp inputs are written (relative to bin/)

# -----------------------------
# PATHS
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/input_Wedge"
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

for M in "${MACH_VALUES[@]}"; do

  # name as in the output convention: always one decimal at least (2 -> 2.0)
  case "$M" in *.*) MNAME="$M" ;; *) MNAME="$M.0" ;; esac

  read STOP PLOTDT DTMAX BRINK ENERGY NOTE < <(
    awk -v M="$M" '
      BEGIN{
        CONVFMT="%.6g"; OFMT="%.6g"
        LU = 6.0 / M; nlu = (M >= 3.0) ? 5 : 6
        stop = nlu * LU
        pdt = stop / 10.0; if (pdt > 0.5) pdt = 0.5
        dtmax = 3.0e-3 / M
        brink = int(400.0 * M / 100.0 + 0.5) * 100
        energy = 1.785714 + 0.5 * M * M
        printf "%.4g %.4g %.3g %d %.6f %d_L/U,_L/U=6/%s=%.3g\n", stop, pdt, dtmax, brink, energy, nlu, M, LU
      }'
  )
  NOTE=${NOTE//_/ }

  OUTPUT_PATH="${OUTPUT_ROOT}/output_Ma${MNAME}"
  INPUT_FILE="${INPUT_DIR}/tmp_Wedge_Ma${MNAME}.in"

  sed -e "s|@MACH@|${MNAME}|g" \
      -e "s|@ENERGY@|${ENERGY}|g" \
      -e "s|@OUTPUT_PATH@|${OUTPUT_PATH}|g" \
      -e "s|@PLOT_DT@|${PLOTDT}|g" \
      -e "s|@DT_MAX@|${DTMAX}|g" \
      -e "s|@STOP_TIME@|${STOP}|g" \
      -e "s|@STOP_NOTE@|${NOTE}|g" \
      -e "s|@BRINKMAN@|${BRINK}|g" \
      "$TEMPLATE" > "$INPUT_FILE"

  echo
  echo "============================================================"
  echo "=== Wedge  Ma ${MNAME}   (stop ${STOP}, plot_dt ${PLOTDT}, dt_max ${DTMAX}, brinkman ${BRINK})"
  echo "============================================================"

  if [ -n "$DRY_RUN" ]; then
    echo "    [dry-run] wrote ${INPUT_FILE} (not running)"
    continue
  fi

  case_start=$SECONDS
  $LAUNCH "$EXEC" "$INPUT_FILE" || echo "  [FAILED] Ma ${MNAME}"
  echo "--- Ma ${MNAME} elapsed: $((SECONDS - case_start)) s"

  if [ -z "$KEEP" ]; then
    rm -f "$INPUT_FILE"
  fi
done

echo
echo "=== WEDGE MACH SWEEP COMPLETE  (total: $((SECONDS - overall_start)) s) ==="
echo "    analyze: python3 tests/FlowWedge/reference/analyze_mach_sweep.py \"${OUTPUT_ROOT}/output_Ma{ma}\""
