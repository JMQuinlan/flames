#!/usr/bin/env bash
# =============================================================================
# build_pele_deps.sh
# Builds SUNDIALS, then PelePhysics, into the ext/ tree so they can be
# linked against the flames2 hydro2 build.
#
# Run from the flames2 repo root:
#     ./scripts/build_pele_deps.sh
#
# After this completes, set in your shell (or in Makefile.local):
#     export SUNDIALS_TARGET=$(pwd)/ext/sundials/install
#     export PELE_PHYSICS_HOME=$(pwd)/ext/pelephysics
#     export PELE_ENABLED=1
#
# Then build flames2 normally (`make hydro2-2d-g++`) and the PelePhysicsEOS
# shim layer will be compiled in.
#
# REQUIREMENTS:
#   - Linux or WSL/macOS (PelePhysics is NOT well-supported on bare Windows;
#     use WSL2 Ubuntu if you're on Windows)
#   - cmake >= 3.20
#   - g++ >= 9.0 (with C++17 support)
#   - Python 3.8+ with numpy (PelePhysics FUEGO mechanism preprocessor)
#   - AMReX already built (you've done this for the main flames2 build)
#
# Tested with: SUNDIALS v7.1.1, PelePhysics master (as of clone date).
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT_DIR="${REPO_ROOT}/ext"

SUNDIALS_SRC="${EXT_DIR}/sundials"
SUNDIALS_BUILD="${EXT_DIR}/sundials/build"
SUNDIALS_INSTALL="${EXT_DIR}/sundials/install"

PELE_SRC="${EXT_DIR}/pelephysics"

if [[ ! -d "${SUNDIALS_SRC}" ]]; then
  echo "ERROR: ${SUNDIALS_SRC} not present."
  echo "Clone with:"
  echo "  git clone --depth=1 --branch v7.1.1 https://github.com/LLNL/sundials.git ${SUNDIALS_SRC}"
  exit 1
fi
if [[ ! -d "${PELE_SRC}" ]]; then
  echo "ERROR: ${PELE_SRC} not present."
  echo "Clone with:"
  echo "  git clone --depth=1 https://github.com/AMReX-Combustion/PelePhysics.git ${PELE_SRC}"
  exit 1
fi

# -----------------------------------------------------------------------------
# 1. Build SUNDIALS (CVODE, ARKODE, NVECTOR_SERIAL).
#    PelePhysics needs CVODE for the chemistry stiff ODE integration.
#    NVECTOR_SERIAL is fine for CPU; add CUDA/HIP backends later if needed.
# -----------------------------------------------------------------------------
echo ">>> Building SUNDIALS at ${SUNDIALS_BUILD}"
mkdir -p "${SUNDIALS_BUILD}"
cd "${SUNDIALS_BUILD}"
cmake -DCMAKE_INSTALL_PREFIX="${SUNDIALS_INSTALL}" \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DBUILD_CVODE=ON \
      -DBUILD_ARKODE=ON \
      -DBUILD_CVODES=OFF \
      -DBUILD_IDA=OFF \
      -DBUILD_IDAS=OFF \
      -DBUILD_KINSOL=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_STATIC_LIBS=ON \
      -DENABLE_MPI=OFF \
      -DEXAMPLES_ENABLE_C=OFF \
      -DEXAMPLES_INSTALL=OFF \
      "${SUNDIALS_SRC}"
make -j"$(nproc 2>/dev/null || echo 4)"
make install
echo ">>> SUNDIALS installed to ${SUNDIALS_INSTALL}"

# -----------------------------------------------------------------------------
# 2. Build PelePhysics.
#    For Phase 2 we want the simplest possible config: GammaLaw EOS,
#    a single-species "air" mechanism, no transport, no reactions.
#    PelePhysics' Make.PelePhysics is GNUMake-based and integrates with
#    AMReX's build conventions, similar to how flames2 already builds AMReX.
#
#    NOTE: PelePhysics expects PELE_PHYSICS_HOME to be set when its
#    Make.PelePhysics is included.  The flames2 Makefile will set this
#    automatically (see Makefile changes in this scaffolding).
# -----------------------------------------------------------------------------
echo ">>> Configuring PelePhysics for GammaLaw + air mechanism"
export PELE_PHYSICS_HOME="${PELE_SRC}"
# The default Mechanism/air mechanism exists in PelePhysics out of the box.
# To use it: set Chemistry_Model = air in your build.
# PelePhysics doesn't build a standalone static library; instead it builds
# directly into the parent application's object tree.  So no `make` step
# here -- the integration happens via Make.PelePhysics being included from
# the parent Makefile.

if [[ -d "${PELE_SRC}/Mechanisms/air" ]]; then
  echo ">>> Default air mechanism present at ${PELE_SRC}/Mechanisms/air"
else
  echo "WARN: ${PELE_SRC}/Mechanisms/air not found; check PelePhysics version."
fi

echo ""
echo "============================================================"
echo "DONE.  Now set in your shell (or in scripts/env_pele.sh):"
echo ""
echo "  export SUNDIALS_TARGET=${SUNDIALS_INSTALL}"
echo "  export PELE_PHYSICS_HOME=${PELE_SRC}"
echo "  export PELE_ENABLED=1"
echo ""
echo "Then rebuild flames2:"
echo "  make hydro2-2d-g++"
echo "============================================================"
