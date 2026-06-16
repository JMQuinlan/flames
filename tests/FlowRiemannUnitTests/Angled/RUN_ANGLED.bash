#!/bin/bash
#
# Run the seven rotated Garrick Riemann tests (theta = 0, 15, 30, 45, 60, 75, 90 deg)
# back-to-back. Resolves paths relative to its own location, so it can be invoked
# from anywhere:
#   bash tests/FlowRiemannUnitTests/Angled/RUN_ANGLED.bash
#   NP=4 bash tests/FlowRiemannUnitTests/Angled/RUN_ANGLED.bash
#
# After everything finishes, run the comparison plot:
#   cd tests/FlowRiemannUnitTests/Angled/reference && python RiemannGarrickCompare.py

set -e

NP=${NP:-8}
EXEC=./hydro2-2d-g++
INPUT_DIR=../tests/FlowRiemannUnitTests/Angled
ANGLES=(00 15 30 45 60 75 90)

# cd to <project_root>/bin regardless of where this script was invoked from
cd "$(dirname "$0")/../../../bin"

if [ ! -x "$EXEC" ]; then
  echo "ERROR: Executable not found or not runnable: $(pwd)/$EXEC"
  exit 1
fi

overall_start=$SECONDS

for angle in "${ANGLES[@]}"; do
  INPUT_FILE="$INPUT_DIR/input_Garrick_${angle}"
  if [ ! -f "$INPUT_FILE" ]; then
    echo "Missing input: $INPUT_FILE"
    exit 1
  fi

  echo
  echo "============================================================"
  echo "=== Garrick Angled  theta = ${angle} deg  (NP=$NP)"
  echo "============================================================"

  case_start=$SECONDS
  mpirun -np "$NP" "$EXEC" "$INPUT_FILE" || true
  echo "--- theta=${angle} elapsed: $((SECONDS - case_start)) s"
done

echo
echo "=== ALL ANGLED RUNS COMPLETE  (total: $((SECONDS - overall_start)) s) ==="
