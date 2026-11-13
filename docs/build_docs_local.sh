#!/bin/bash
# Build the Sphinx docs with the LOCAL tools (no system install): docs/.venv (Sphinx extensions) and docs/.tools/doxygen
# (Ubuntu doxygen package unpacked with dpkg -x) and docs/.tools/lib/libclang.so -> the system libclang-21.  Usage: bash docs/build_docs_local.sh   -> docs/build/html/index.html
D=$(cd "$(dirname "$0")" && pwd)
export PATH="$D/.tools/doxygen/usr/bin:$D/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$D/.tools/lib:$D/.tools/doxygen/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
make -C "$D" html SPHINXBUILD="$D/.venv/bin/python -m sphinx"
