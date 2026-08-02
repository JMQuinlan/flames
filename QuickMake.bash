#!/bin/bash

# Quickly make both 2d and 3d tests
# Usage: ./QuickMake.bash [ncores]   (default 12)

NCORES=${1:-12}

./configure --dim=2 --get-eigen && make -j"$NCORES" && ./configure --dim=3 --get-eigen && make -j"$NCORES"
