#!/bin/bash
# Desktop sweep for the dilatational-viscosity unit tests.
#   run_sweep.sh EXE OUTROOT [NP] [JOBS] [which] [quick]
#     which = pulse (single phase, default) | water | norm (multiphase)
#     quick = only R0/dx 16 at eps/dx 2
# Every run writes its deck overrides to OUTROOT/<name>/cmdline.txt so the
# analysis scripts can rebuild the parameters.  Runs are 2D and small
# (R0/dx 64 is 384^2 cells); JOBS runs of NP ranks each go at once.
EXE=${1:?2D hydro2 executable}; OUT=${2:?output root}; NP=${3:-1}; JOBS=${4:-4}; WHICH=${5:-pulse}; QUICK=${6:-}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
case $WHICH in
  pulse) DECK=$ROOT/tests/FlowDilationViscosity/input_SinglePhase_Pulse_2D ;;
  water) DECK=$ROOT/tests/FlowDilationViscosity/input_Multiphase_WaterAir_2D; R0=2.0e-6; KON=3.6e-8 ;;
  norm)  DECK=$ROOT/tests/FlowDilationViscosity/input_Multiphase_Normalized_2D; R0=1.0; KON=1.0 ;;
esac
mkdir -p "$OUT"
job() {  # name overrides...
  local n=$1; shift
  mkdir -p "$OUT/$n"; echo "$@" > "$OUT/$n/cmdline.txt"
  ( cd "$ROOT" && timeout 7200 mpirun --bind-to none -np $NP "$EXE" "$DECK" plot_file="$OUT/$n" "$@" \
      < /dev/null > "$OUT/$n.log" 2>&1; echo "$n exit=$?" >> "$OUT/status.txt" )
}
export -f job; export OUT NP EXE ROOT DECK
list() {
  if [ $WHICH = pulse ]; then
    for N in ${PULSE_N:-16 32}; do            # R0/dx (uniform mesh); PULSE_N="16 32 64" for the full set
      for E in 1 2 4; do                      # eps/dx
        [ -n "$QUICK" ] && { [ $N != 16 ] || [ $E != 2 ]; } && continue
        nc=$((6 * N)); eps=$(python3 -c "print($E/$N)")
        for K in 1.0 0.0; do
          echo "P_N${N}_E${E}_K${K%.*} amr.n_cell=$nc $nc eta.ic.expression.constant.epsilon=$eps shell.kappa_s=$K"
        done
      done
    done
  else                                        # multiphase: AMR, base dx = R0/4
    for L in 1 2 3; do                        # R0/dx = 8, 16, 32
      for E in 1 2 4; do
        [ -n "$QUICK" ] && { [ $L != 2 ] || [ $E != 2 ]; } && continue
        eps=$(python3 -c "print($E*$R0/(4*2**$L))")
        for K in $KON 0.0; do
          tag=K1; [ $K = 0.0 ] && tag=K0
          echo "${WHICH}_L${L}_E${E}_${tag} amr.max_level=$L eta.ic.expression.constant.epsilon=$eps shell.kappa_s=$K"
        done
      done
    done
  fi
}
list | xargs -P "$JOBS" -L 1 bash -c 'job "$@"' _
echo ALLDONE >> "$OUT/status.txt"
