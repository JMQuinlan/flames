#!/bin/bash
#
# Run all 20 angled-3D Garrick Riemann tests (4 planes x 5 angles) back-to-back.
# Resolves paths relative to its own location, so invoke from anywhere:
#   bash tests/FlowRiemannUnitTests/Angled_3D/RUN_ANGLED_3D.bash
#   NP=8 bash tests/FlowRiemannUnitTests/Angled_3D/RUN_ANGLED_3D.bash
#
# After everything finishes, overlay the profiles:
#   cd tests/FlowRiemannUnitTests/Angled_3D/reference && python RiemannGarrickCompare_3D.py

set -e

NP=${NP:-8}
EXEC=./hydro2-3d-g++
INPUT_DIR=../tests/FlowRiemannUnitTests/Angled_3D
PLANES=(XY YZ XZ XYZ)
ANGLES=(00 30 45 60 90)

# cd to <project_root>/bin regardless of where this script was invoked from
cd "$(dirname "$0")/../../../bin"

if [ ! -x "$EXEC" ]; then
  echo "ERROR: Executable not found or not runnable: $(pwd)/$EXEC"
  exit 1
fi

overall_start=$SECONDS

for plane in "${PLANES[@]}"; do
  for angle in "${ANGLES[@]}"; do
    INPUT_FILE="$INPUT_DIR/input_Garrick_${plane}_${angle}"
    if [ ! -f "$INPUT_FILE" ]; then
      echo "Missing input: $INPUT_FILE"
      exit 1
    fi

    echo
    echo "============================================================"
    echo "=== Garrick Angled 3D   plane=${plane}  theta=${angle} deg   (NP=$NP)"
    echo "============================================================"

    case_start=$SECONDS
    mpirun -np "$NP" "$EXEC" "$INPUT_FILE" || true
    echo "--- ${plane}_${angle} elapsed: $((SECONDS - case_start)) s"
  done
done

echo
echo "=== ALL 20 ANGLED-3D RUNS COMPLETE  (total: $((SECONDS - overall_start)) s) ==="
