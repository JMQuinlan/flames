#!/bin/bash
#
# NACA 4-digit angle-of-attack sweep from a single template (same pattern as
# tests/FlowWedge/TEMPLATES/sweep_mach.bash and FlowLaplace/TEMPLATES/RUN_LAPLACE.bash).
# For every AoA in AOA_VALUES:
#   1. TEMPLATES/make_naca0012_bmp.py rotates the airfoil and writes phi.bmp
#      (and prints the bitmap bounds + rotated pressure probes)
#   2. the Mach/Re/AoA-dependent values are computed here (awk) and substituted
#      into TEMPLATES/input_NACA0012
#   3. the case is run (JOBS cases at a time)
# Resolves paths relative to its own location, so it can be invoked from anywhere:
#
#   bash tests/FlowAirfoil/TEMPLATES/sweep_aoa.bash
#   AOA="0 4 8" bash tests/FlowAirfoil/TEMPLATES/sweep_aoa.bash      # override the AoA list
#   AIRFOIL=0008 bash ...                                            # any NACA 4-digit section
#   MACH=0.3 RE=5000 MAX_LEVEL=4 bash ...                             # flow / resolution
#   DOM_LO="-20 -30" DOM_HI="40 30" N_CELL="480 480" MAX_LEVEL=5 bash ...  # larger domain
#   NP=4 JOBS=5 bash ...                                              # 5 cases x 4 ranks at a time
#   LIMITER=weno5 bash ...                                            # reconstruction (default vanleer)
#   DRY_RUN=1 bash ...                                                # write inputs + bitmaps, don't run
#   OUTPUT_ROOT=/mmfs1/home/ttryon/flames/bin/tests/FlowAirfoil/<name> \
#   LAUNCH="srun -n 128 --mpi=pmi2" bash ...                         # INCLINE
#
# Each case lands in  $OUTPUT_ROOT/aoa_<AoA>/  (input, phi.bmp, out/ plotfiles,
# out_forces.dat per-step force + probe history, run.log).  Afterwards:
#   python3 tests/FlowAirfoil/reference/analyze_aoa_sweep.py <OUTPUT_ROOT>
#
# Per-case rules (freestream rho = 1, p = 1/gamma so c = 1 and U = Mach; chord 1):
#   mu        = rho U c / Re = MACH / RE
#   AoA = 0:  refine ONLY by the mirror-symmetric refine_box (phi / p / rho
#             criteria off) -- AMReX clustering is not mirror-symmetric and
#             gradient tags broke the symmetry (spurious lift at AoA 0)
#   AoA != 0: phi 0.1 and p/rho 0.2 gradient refinement, and a refine_box that
#             grows with |AoA| to keep the tilted wake on the finest level:
#               lo = (-0.65, -0.2 - 0.01 max(AoA,0))
#               hi = (1.2 + 0.015 |AoA|, 0.2 + 0.01 max(-AoA,0) + 0.0025 |AoA|)
#   finest dx = (xhi-xlo)/Nx/2^MAX_LEVEL  (default domain [-5,11] x [-6,6], base 256 x 192)
#   DOM_LO="-20 -30" DOM_HI="40 30" N_CELL="480 480" MAX_LEVEL=5 -> same finest dx, 20 c
#   upstream / 40 c downstream / +-30 c lateral

set -e

# -----------------------------
# INPUTS
# -----------------------------
AOA_VALUES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
[ -n "$AOA" ] && read -r -a AOA_VALUES <<< "$AOA"

AIRFOIL=${AIRFOIL:-0012}               # NACA 4-digit code (0012, 0008, 4402, ...)
MACH=${MACH:-0.3}
RE=${RE:-5000}
MAX_LEVEL=${MAX_LEVEL:-4}
STOP_TIME=${STOP_TIME:-30.0}          # 9 chord flow-through times at M 0.3 (c/U = 3.33)
PLOT_DT=${PLOT_DT:-0.5}
DOM_LO=${DOM_LO:-"-5.0 -6.0"}         # domain (chord 1, airfoil centred at the origin)
DOM_HI=${DOM_HI:-"11.0 6.0"}
N_CELL=${N_CELL:-"256 192"}           # base grid; finest dx = (xhi-xlo)/Nx/2^MAX_LEVEL
LIMITER=${LIMITER:-vanleer}           # Limiter.type: vanleer | minmod | weno3 | weno5 (template nghost = 4)

NP=${NP:-4}
JOBS=${JOBS:-1}
EXEC=${EXEC:-./hydro2-2d-g++}
LAUNCH=${LAUNCH:-mpirun -np $NP --bind-to none}
OUTPUT_ROOT=${OUTPUT_ROOT:-./tests/FlowAirfoil/NACA${AIRFOIL}_AoA_sweep_M${MACH}_Re${RE}_L${MAX_LEVEL}}

# -----------------------------
# PATHS
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/input_NACA0012"
BMPGEN="$SCRIPT_DIR/make_naca0012_bmp.py"
BIN_DIR="$(cd "$SCRIPT_DIR/../../../bin" && pwd)"

for f in "$TEMPLATE" "$BMPGEN"; do
  [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
cd "$BIN_DIR"
if [ -z "$DRY_RUN" ] && [ ! -x "$EXEC" ]; then
  echo "ERROR: executable not found or not runnable: $(pwd)/$EXEC  (DRY_RUN=1 to only write inputs)"
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

# -----------------------------
# ONE CASE
# -----------------------------
run_case() {
  local A="$1"
  local ANAME; ANAME=$(awk -v a="$A" 'BEGIN{printf "%04.1f", a}')
  local D="$OUTPUT_ROOT/aoa_${ANAME}"
  mkdir -p "$D"

  # rotated bitmap + bounds + probes
  read BLX BLY BHX BHY PX1 PX2 PX3 PX4 PY1 PY2 PY3 PY4 < <(python3 "$BMPGEN" "$A" "$D/phi.bmp" "$AIRFOIL")

  read MU DXF PHIC GRADC RLO RHI < <(
    awk -v M="$MACH" -v R="$RE" -v L="$MAX_LEVEL" -v A="$A" -v DL="$DOM_LO" -v DH="$DOM_HI" -v NC="$N_CELL" '
      BEGIN{
        split(DL, lo, " "); split(DH, hi, " "); split(NC, nc, " ")
        mu = M / R; dxf = (hi[1] - lo[1]) / nc[1] / (2 ^ L)
        aa = (A < 0) ? -A : A; ap = (A > 0) ? A : 0; an = (A < 0) ? -A : 0
        if (A == 0) { phic = "1e10"; gradc = "1e10" } else { phic = "0.1"; gradc = "0.2" }
        printf "%.6e %.5f %s %s %.3f_%.3f %.3f_%.3f\n", mu, dxf, phic, gradc,
               -0.65, -0.2 - 0.01 * ap, 1.2 + 0.015 * aa, 0.2 + 0.01 * an + 0.0025 * aa
      }')
  RLO=${RLO//_/ }; RHI=${RHI//_/ }

  sed -e "s|@AIRFOIL@|${AIRFOIL}|g" \
      -e "s|@MACH@|${MACH}|g" \
      -e "s|@RE@|${RE}|g" \
      -e "s|@AOA@|${A}|g" \
      -e "s|@MU@|${MU}|g" \
      -e "s|@STOP_TIME@|${STOP_TIME}|g" \
      -e "s|@PLOT_DT@|${PLOT_DT}|g" \
      -e "s|@MAX_LEVEL@|${MAX_LEVEL}|g" \
      -e "s|@DOM_LO@|${DOM_LO}|g" \
      -e "s|@DOM_HI@|${DOM_HI}|g" \
      -e "s|@N_CELL@|${N_CELL}|g" \
      -e "s|@DXF@|${DXF}|g" \
      -e "s|@OUTPUT_PATH@|${D}/out|g" \
      -e "s|@BMP_PATH@|${D}/phi.bmp|g" \
      -e "s|@BMP_LO@|${BLX} ${BLY}|g" \
      -e "s|@BMP_HI@|${BHX} ${BHY}|g" \
      -e "s|@PROBE_X@|${PX1} ${PX2} ${PX3} ${PX4}|g" \
      -e "s|@PROBE_Y@|${PY1} ${PY2} ${PY3} ${PY4}|g" \
      -e "s|@RBOX_LO@|${RLO}|g" \
      -e "s|@RBOX_HI@|${RHI}|g" \
      -e "s|@PHI_CRIT@|${PHIC}|g" \
      -e "s|@GRAD_CRIT@|${GRADC}|g" \
      -e "s|@LIMITER@|${LIMITER}|g" \
      "$TEMPLATE" > "$D/input"

  echo "=== $(date '+%F %T')  NACA ${AIRFOIL}  AoA ${A}  M ${MACH}  Re ${RE}  L${MAX_LEVEL}  ${LIMITER}  mu ${MU}  -> ${D}"
  if [ -n "$DRY_RUN" ]; then
    echo "    [dry-run] wrote ${D}/input and phi.bmp"
    return 0
  fi
  local t0=$SECONDS
  $LAUNCH "$EXEC" "$D/input" > "$D/run.log" 2>&1 < /dev/null || echo "  [FAILED] AoA ${A} (see ${D}/run.log)"
  echo "--- $(date '+%F %T')  AoA ${A} done, elapsed $((SECONDS - t0)) s"
}

# -----------------------------
# MAIN LOOP  (JOBS cases concurrently)
# -----------------------------
overall_start=$SECONDS
for A in "${AOA_VALUES[@]}"; do
  if [ "$JOBS" -gt 1 ]; then
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    run_case "$A" &
    sleep 5
  else
    run_case "$A"
  fi
done
wait

echo
echo "=== AOA SWEEP COMPLETE  (total: $((SECONDS - overall_start)) s)  ->  ${OUTPUT_ROOT}"
echo "    analyze: python3 tests/FlowAirfoil/reference/analyze_aoa_sweep.py ${OUTPUT_ROOT}"
