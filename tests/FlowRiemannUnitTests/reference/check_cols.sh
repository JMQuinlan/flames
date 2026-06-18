#!/bin/bash
# Extract ALL variables (no -v) so we can read the true column order from the
# header, then print far-left / center / far-right rows for the IC and evolved
# Garrick_x state.  fextract segfaults on exit (harmless); no set -e.
FX=ext/AMReX-Codes/amrex/Tools/Plotfile/fextract.gnu.ex

for tag in IC EVOLVED; do
  if [ "$tag" = IC ]; then PF=tests/FlowRiemannUnitTests/output_Garrick_x/00000cell; fi
  if [ "$tag" = EVOLVED ]; then PF=tests/FlowRiemannUnitTests/output_Garrick_x/00100cell; fi
  $FX -d 0 -s /tmp/all.txt "$PF" 2>/dev/null
  echo "===== $tag ($PF) ====="
  echo "--- variable column header ---"
  grep -E '^#' /tmp/all.txt | tail -1
  echo "--- far-left row ---"
  grep -vE '^#|^$' /tmp/all.txt | sed -n '1p'
  echo "--- far-right row ---"
  grep -vE '^#|^$' /tmp/all.txt | sed -n '800p'
done
