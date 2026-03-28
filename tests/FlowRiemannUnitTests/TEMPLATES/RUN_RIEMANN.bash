#!/bin/bash

set -e  # Exit on any error

# -----------------------------
# INPUTS
# -----------------------------

# Solvers
SOLVER_LABELS=(
  Roe
  HLLE
  HLLC
  HLLC_All_Mach
  HLLC_All_Mach_Furfaro
  HLLC_Jaimar_Ooman
)

# Test cases
TESTS=(
  Toro1a
  Toro1b
  Toro1r
  Toro2
  Toro3
  Toro4
  Toro4b
  Toro5
  Toro6
  Toro7
)

# Epsilon values for diffuse interface
EPSILONS=("1e-1" "1e-2" "1e-3")

NP=8
EXEC=./hydro2-2d-g++
OUTDIR=./tests/FlowRiemannUnitTestsAUTO
TEMPLATE_DIR=../tests/FlowRiemannUnitTests/TEMPLATES  # folder containing test-specific templates

# -----------------------------
# MAIN LOOP
# -----------------------------

for test in "${TESTS[@]}"; do

  # Determine base test name (if needed for template, e.g., Toro1a -> Toro1)
  # If each test has its own template, skip this line
  # base_test=$(echo "$test" | sed 's/[a-z]$//')
  base_test=$test

  TEMPLATE_SHARP="${TEMPLATE_DIR}/input_${base_test}_shrp"
  TEMPLATE_DIFFUSE="${TEMPLATE_DIR}/input_${base_test}_diff"

  # Checking to see if file exsists
  if [ ! -f "$TEMPLATE_SHARP" ]; then
    echo "Missing template: $TEMPLATE_SHARP"
    exit 1
  fi
  if [ ! -f "$TEMPLATE_DIFFUSE" ]; then
    echo "Missing template: $TEMPLATE_DIFFUSE"
    exit 1
  fi

  for solver_label in "${SOLVER_LABELS[@]}"; do
    # Convert to lowercase to call riemann class
    solver=${solver_label,,}

    # =============================
    # SHARP CASE
    # =============================
    echo "=== Running $test | $solver_label | SHARP ==="

    INPUT_FILE="tmp_${test}_${solver}_sharp.in"
    OUTPUT_PATH="${OUTDIR}/output_${test}_${solver_label}_Sharp"

    sed -e "s|SOLVER_NAME|$solver|g" \
        -e "s|OUTPUT_PATH|$OUTPUT_PATH|g" \
        "$TEMPLATE_SHARP" > "$INPUT_FILE"

    mpirun -np $NP $EXEC "$INPUT_FILE" # Run shrp case
    rm "$INPUT_FILE" # Remove file to clean work space

    # =============================
    # DIFFUSE CASES
    # =============================
    for eps in "${EPSILONS[@]}"; do
      echo "=== Running $test | $solver_label | DIFFUSE | eps=$eps ==="

      INPUT_FILE="tmp_${test}_${solver}_diff_eps${eps}.in"
      OUTPUT_PATH="${OUTDIR}/output_${test}_${solver_label}_Diffuse_eps${eps}"

      sed -e "s|SOLVER_NAME|$solver|g" \
          -e "s|OUTPUT_PATH|$OUTPUT_PATH|g" \
          -e "s|EPSILON|$eps|g" \
          "$TEMPLATE_DIFFUSE" > "$INPUT_FILE"

      mpirun -np $NP $EXEC "$INPUT_FILE" # Run diff case
      rm "$INPUT_FILE" # Remove file to clean work space

    done

  done
done

echo "=== ALL RUNS COMPLETE ==="
