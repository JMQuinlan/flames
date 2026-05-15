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

# Time integration schemes (AMReX_RKIntegrator.H tableau types).
# Format used in input substitution:
#   integration.type    = ${INTEGRATION_TYPE}
#   integration.rk.type = ${INTEGRATION_RK_TYPE}
# For ForwardEuler the rk.type is ignored by AMReX (we still set it to a
# dummy 1 so the placeholder is always replaced).
INTEGRATION_LABELS=(
  ForwardEuler
  Trapezoid
  SSPRK3
  RK4
)

# Map label -> (integration.type, integration.rk.type)
declare -A INT_TYPE
declare -A INT_RK_TYPE
INT_TYPE[ForwardEuler]=ForwardEuler;   INT_RK_TYPE[ForwardEuler]=1
INT_TYPE[Trapezoid]=RungeKutta;        INT_RK_TYPE[Trapezoid]=2
INT_TYPE[SSPRK3]=RungeKutta;           INT_RK_TYPE[SSPRK3]=3
INT_TYPE[RK4]=RungeKutta;              INT_RK_TYPE[RK4]=4

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

  base_test=$test

  TEMPLATE_SHARP="${TEMPLATE_DIR}/input_${base_test}_shrp"
  TEMPLATE_DIFFUSE="${TEMPLATE_DIR}/input_${base_test}_diff"

  if [ ! -f "$TEMPLATE_SHARP" ]; then
    echo "Missing template: $TEMPLATE_SHARP"
    exit 1
  fi
  if [ ! -f "$TEMPLATE_DIFFUSE" ]; then
    echo "Missing template: $TEMPLATE_DIFFUSE"
    exit 1
  fi

  for solver_label in "${SOLVER_LABELS[@]}"; do
    solver=${solver_label,,}

    for limiter_label in "${LIMITER_LABELS[@]}"; do
      limiter=${limiter_label,,}

      for integration_label in "${INTEGRATION_LABELS[@]}"; do
        integration_type="${INT_TYPE[$integration_label]}"
        integration_rk_type="${INT_RK_TYPE[$integration_label]}"

        # =============================
        # SHARP CASE
        # =============================
        echo "=== Running $test | $solver_label | $limiter_label | $integration_label | SHARP ==="

        INPUT_FILE="tmp_${test}_${solver}_${limiter}_${integration_label,,}_sharp.in"
        OUTPUT_PATH="${OUTDIR}/${test}/output_${test}_${solver_label}_${limiter_label}_${integration_label}_Sharp_Interface"

        sed -e "s|SOLVER_NAME|$solver|g" \
            -e "s|LIMITER_NAME|$limiter|g" \
            -e "s|INTEGRATION_TYPE|$integration_type|g" \
            -e "s|INTEGRATION_RK_TYPE|$integration_rk_type|g" \
            -e "s|OUTPUT_PATH|$OUTPUT_PATH|g" \
            "$TEMPLATE_SHARP" > "$INPUT_FILE"

        mpirun -np $NP $EXEC "$INPUT_FILE" || true # Run shrp case
        rm "$INPUT_FILE"

        # =============================
        # DIFFUSE CASES
        # =============================
        for eps in "${EPSILONS[@]}"; do
          echo "=== Running $test | $solver_label | $limiter_label | $integration_label | DIFFUSE | eps=$eps ==="

          INPUT_FILE="tmp_${test}_${solver}_${limiter}_${integration_label,,}_diff_eps${eps}.in"
          OUTPUT_PATH="${OUTDIR}/${test}/output_${test}_${solver_label}_${limiter_label}_${integration_label}_epsilon_${eps}"

          sed -e "s|SOLVER_NAME|$solver|g" \
              -e "s|LIMITER_NAME|$limiter|g" \
              -e "s|INTEGRATION_TYPE|$integration_type|g" \
              -e "s|INTEGRATION_RK_TYPE|$integration_rk_type|g" \
              -e "s|OUTPUT_PATH|$OUTPUT_PATH|g" \
              -e "s|EPSILON|$eps|g" \
              "$TEMPLATE_DIFFUSE" > "$INPUT_FILE"

          mpirun -np $NP $EXEC "$INPUT_FILE" || true
          rm "$INPUT_FILE"

        done

      done
    done
  done
done

echo "=== ALL RUNS COMPLETE ==="
