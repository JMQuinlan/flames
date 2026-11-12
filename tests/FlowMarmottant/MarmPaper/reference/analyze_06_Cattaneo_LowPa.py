#!/usr/bin/env python3
"""
MarmPaper 06 VALIDATION, low amplitude: coated microbubble (R0 = 2.95 um) in the measured
1 MHz pulse at p_a = 60 kPa, against Cattaneo et al., Nature Physics 21:590 (2025), Fig. 2a
(deck ../input_06_Cattaneo_LowPa).

Figure <figdir>/<stem>_R.{png,eps}: R [um] vs t [us] (t = 0 at the start of the burst)
  (a) the whole run, (b) zoom on the first large cycles (shaded in (a)), with
    coated Keller-Miksis and coated Rayleigh-Plesset (Marmottant shell, polytropic gas, the run's
        kappa_s / Cross law, measured pulse envelope),
    the Cattaneo et al. model (Marmottant shell + heat-conducting Zhou gas, constant kappa_s),
    the experiment (open markers; figure time shifted by -1.57 us onto the simulation axis),
    the simulation volumetric radius R_vol and the eta = 0.5 radius.
Printed: peak and trough radius in each drive cycle for every curve, with the % gap vs experiment.
Model parameters come from the deck, overridden by <output>/metadata (command-line overrides of the
run, e.g. shell.kappa_s_alpha=0).  RADIAL slabs (n_cell N x 2 x 2) and AMR octants are both read
(detected from the plotfiles; see cattaneo_06_common.py).

    python3 analyze_06_Cattaneo_LowPa.py                                   # deck plot_file
    python3 analyze_06_Cattaneo_LowPa.py --output RUN_DIR --figdir FIG_DIR
    python3 analyze_06_Cattaneo_LowPa.py --output RUN_DIR --zoom 1.5 4.5 --rvol-norm --thermo
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import cattaneo_06_common as cc          # noqa: E402

if __name__ == "__main__":
    cc.main(case_tag="lo",
            deck=os.path.normpath(os.path.join(_HERE, "..", "input_06_Cattaneo_LowPa")),
            default_output="/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper/output_06_Cattaneo_LowPa",
            stem="06_Cattaneo_LowPa", zoom_default=[1.5, 4.5])
