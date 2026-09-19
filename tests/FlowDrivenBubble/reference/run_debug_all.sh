#!/bin/bash
# Run driven_debug.py over every driven-bubble case that has frames on disk.
# Safe on RUNNING jobs: frames are read newest-last and a half-written
# plotfile is skipped with a warning.
#
#   ./run_debug_all.sh                 # every input_* in ../
#   ./run_debug_all.sh input_f25.0_A1.00 input_f81.5_A1.36
#   EVERY=4 ./run_debug_all.sh         # subsample frames (faster)
#   OUTROOT=/path/to/plotfile/parent ./run_debug_all.sh
set -u
EVERY=${EVERY:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

inputs=("$@")
[ ${#inputs[@]} -eq 0 ] && inputs=(../input_*)

for f in "${inputs[@]}"; do
    [ -f "$f" ] || f="../$f"
    [ -f "$f" ] || { echo "SKIP  (no such input) $f"; continue; }
    name=$(basename "$f")
    case "$name" in *.sh|*.py|*~) continue;; esac

    # where are the plotfiles?
    pf=$(grep -E '^[[:space:]]*plot_file' "$f" | tail -1 | cut -d= -f2 | tr -d ' ')
    dir=""
    for c in "${OUTROOT:-}${OUTROOT:+/}$(basename "$pf")" "$pf" \
             "../../../bin/tests/FlowDrivenBubble/$(basename "$pf")" \
             "../../../bin/tests/FlowDrivenBubble/domaintest/$(basename "$pf")"; do
        [ -n "$c" ] && [ -d "$c" ] && { dir="$c"; break; }
    done
    if [ -z "$dir" ]; then echo "SKIP  (no output dir) $name"; continue; fi

    n=$(find -L "$dir" -maxdepth 1 -name '*cell' -type d 2>/dev/null | wc -l)
    if [ "$n" -lt 3 ]; then echo "SKIP  ($n frames, too few) $name"; continue; fi

    echo "=== $name  ($n frames)  ==="
    python3 driven_debug.py --input "$f" --output "$dir" --every "$EVERY" \
        || echo "FAILED $name"
done

echo
echo "Reports : $HERE/driven_debug_*.txt"
echo "Data    : $HERE/driven_debug_*.npz   <-- send these back"
echo "Figures : $HERE/Images/driven_debug_*.png"
