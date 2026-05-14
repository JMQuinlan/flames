#!/bin/bash

set -e  # Exit on any error

# -----------------------------
# INPUTS
# -----------------------------

# Riemann Solvers
SOLVER_LABELS=(
  Roe
  HLLE
  HLLC
  HLLC_Oomar_Jaiman
  HLLC_All_Mach
  HLLC_All_Mach_Furfaro
)

# Limiters / spatial reconstruction
# (godunov = 1st-order baseline; minmod/vanleer = MUSCL2;
#  weno3/weno5 = high-order non-oscillatory)
LIMITER_LABELS=(
  Godunov
  Minmod
  VanLeer
  WENO3
  WENO5
)

# Test cases
TESTS=(
  Toro1a
  Toro1b
  Toro1r
  Toro2
  Toro3
  Toro4a
  Toro4b
  Toro5
  Toro6
  Toro7
  Garrick
)

# Epsilon values for diffuse interface
EPSILONS=("1e-1" "1e-2" "1e-3" "1e-4")

NP=6
EXEC=./hydro2-2d-g++
OUTDIR=./tests/FlowRiemannUnitTestsAUTO
TEMPLATE_DIR=../tests/FlowRiemannUnitTests/TEMPLATES  # folder containing test-specific templates

# -----------------------------
# MAIN LOOP
# -----------------------------

for test in "${TESTS[@]}"; do

  # Determine base test name (if needed for template, e.g., Toro1a -> Toro1)
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

    for limiter_label in "${LIMITER_LABELS[@]}"; do
      # Convert to lowercase to call limiter class (matches name= in C++)
      limiter=${limiter_label,,}

      # =============================
      # SHARP CASE
      # =============================
      echo "=== Running $test | $solver_label | $limiter_label | SHARP ==="

      INPUT_FILE="tmp_${test}_${solver}_${limiter}_sharp.in"
      OUTPUT_PATH="${OUTDIR}/${test}/output_${test}_${solver_label}_${limiter_label}_Sharp_Interface"

      sed -e "s|SOLVER_NAME|$solver|g" \
          -e "s|LIMITER_NAME|$limiter|g" \
          -e "s|OUTPUT_PATH|$OUTPUT_PATH|g" \
          "$TEMPLATE_SHARP" > "$INPUT_FILE"

      mpirun -np $NP $EXEC "$INPUT_FILE" || true # Run shrp case
      rm "$INPUT_FILE" # Remove file to clean work space

      # =============================
      # DIFFUSE CASES
      # =============================
      for eps in "${EPSILONS[@]}"; do
        echo "=== Running $test | $solver_label | $limiter_label | DIFFUSE | eps=$eps ==="

        INPUT_FILE="tmp_${test}_${solver}_${limiter}_diff_eps${eps}.in"
        OUTPUT_PATH="${OUTDIR}/${test}/output_${test}_${solver_label}_${limiter_label}_epsilon_${eps}"

        sed -e "s|SOLVER_NAME|$solver|g" \
            -e "s|LIMITER_NAME|$limiter|g" \
            -e "s|OUTPUT_PATH|$OUTPUT_PATH|g" \
            -e "s|EPSILON|$eps|g" \
            "$TEMPLATE_DIFFUSE" > "$INPUT_FILE"

        mpirun -np $NP $EXEC "$INPUT_FILE" || true # Run diff case
        rm "$INPUT_FILE" # Remove file to clean work space

      done

    done

  done
done

echo "=== ALL RUNS COMPLETE ==="
