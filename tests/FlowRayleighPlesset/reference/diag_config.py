"""Shared config for the FlowRayleighPlesset diagnostic toolkit.

Every diagnostic tool here resolves its plotfile directory the same way:
    python <tool>.py [output_dir]
    - if output_dir is given on the command line, use it
    - otherwise fall back to DEFAULT_OUTPUT_DIR below

Edit DEFAULT_OUTPUT_DIR (or the _CANDIDATES list) to point at your run.  The first
existing candidate wins, so the SAME config works on the cluster and locally.
"""
import os
import sys

R0 = 0.02          # bubble initial radius [m]  (match eta.ic R0 in your input)
N_BASE = 160       # base AMR cells per side    (dx_finest fallback only)
MAX_LEVEL = 4      # max AMR level              (dx_finest fallback only)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    "/mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Oscillating_Large_3D",
    os.path.normpath(os.path.join(_HERE, "..", "output_Sch20_Oscillating_Large_3D")),
    os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "FLAMES_Out")),
]
DEFAULT_OUTPUT_DIR = next((c for c in _CANDIDATES if os.path.isdir(c)), _CANDIDATES[0])


def resolve_dir(argv=None):
    """Directory from argv[1] if present, else DEFAULT_OUTPUT_DIR."""
    argv = argv if argv is not None else sys.argv
    return argv[1] if len(argv) > 1 else DEFAULT_OUTPUT_DIR
