#!/bin/bash
# Submit all ten 3D Marmottant Laplace cases.  Usage: ./SubmitAll3D.sh [--dry]
cd "$(dirname "$0")"
for s in Run3D_R*.sh; do
  if [ "${1:-}" = "--dry" ]; then echo "sbatch $s"; else sbatch "$s"; fi
done
