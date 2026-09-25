#!/usr/bin/env python3
"""Run compare_wedge_angles.py against the local desktop outputs instead of the
INCLINE /mmfs1 path, without editing the analysis scripts.

  python3 run_compare_local.py                       # bin/tests/FlowWedge/output_Ma{ma} -> ./Images
  python3 run_compare_local.py <output_tmpl> <img>   # e.g. .../newdefaults/Ma{ma}/output  ./Images/newdefaults
"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here); os.chdir(here)
import wedge_analysis as wa
tmpl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "../../../bin/tests/FlowWedge/output_Ma{ma}")
wa.OUTPUT_TMPL = os.path.normpath(os.path.abspath(tmpl))
import compare_wedge_angles as cwa
if len(sys.argv) > 2:
    wa.OUTPUT_DIR = cwa.OUTPUT_DIR = sys.argv[2]
cwa.main()
