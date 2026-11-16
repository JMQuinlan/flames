#!/usr/bin/env python3
"""
MarmPaper 06 VALIDATION, high amplitude: coated microbubble (R0 = 2.95 um) in the measured
1 MHz pulse at p_a = 160 kPa, against Cattaneo et al., Nature Physics 21:590 (2025), Fig. 2a
(deck ../input_06_Cattaneo_HighPa).

Figure <figdir>/<stem>_R.{png,eps}: R [um] vs t [us] (t = 0 at the start of the burst)
  (a) the whole run, (b) zoom on the first large cycles (shaded in (a)), with
    coated Keller-Miksis and coated Rayleigh-Plesset (Marmottant shell, polytropic gas, the run's
        kappa_s / Cross law, measured pulse envelope),
    the Cattaneo et al. model (Marmottant shell + heat-conducting Zhou gas, constant kappa_s),
    the experiment (filled markers; figure time shifted by -1.57 us onto the simulation axis),
    the simulation volumetric radius R_vol and the eta = 0.5 radius.
Figure <figdir>/<stem>_R_full.{png,eps}: panel (a) only (compact).
Figure <figdir>/<stem>_thermo.{png,eps} (default; --no-thermo-fig skips it): mean gas pressure / p0 vs time on the
  Fig. 2a time axis (t + 1.57 us) from thermo.dat, the present model against isothermal (p V = const) and adiabatic
  (p V^gamma = const) compression along the simulated radius and the Cattaneo et al. heat-conducting-gas model;
  prints the effective polytropic exponent per compression.  All figures are written as .png and .eps (--formats).
Printed: peak and trough radius in each drive cycle for every curve, with the % gap vs experiment.
Model parameters come from the deck, overridden by <output>/metadata (command-line overrides of the
run, e.g. shell.kappa_s_alpha=0).  RADIAL slabs (n_cell N x 2 x 2) and AMR octants are both read
(detected from the plotfiles; see cattaneo_06_common.py).

    python3 analyze_06_Cattaneo_HighPa.py                                   # deck plot_file
    python3 analyze_06_Cattaneo_HighPa.py --output RUN_DIR --figdir FIG_DIR
    python3 analyze_06_Cattaneo_HighPa.py --output RUN_DIR --zoom 1.5 4.5 --rvol-norm --thermo

Animated GIFs (cattaneo_06_gif.py; --gif adds them to the static figure, --gif-only skips the latter):
<stem>_bubble.gif (cross-section through the bubble centre, p - p_inf on a blue-red symlog scale,
eta = 0.5 contour), <stem>_bubble_R.gif (same + R(t) with experiment / Cattaneo model / simulation),
<stem>_bubble3d.gif (--gif-3d, eta = 0.5 isosurface; octant / full).  Time axis: Fig. 2a (default) or sim.
    python3 analyze_06_Cattaneo_HighPa.py --output RUN_DIR --figdir FIG_DIR --gif --gif-every 2 --gif-fps 10
    python3 analyze_06_Cattaneo_HighPa.py --output RUN_DIR --gif-only --gif-3d --gif-time sim
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import cattaneo_06_common as cc          # noqa: E402

if __name__ == "__main__":
    cc.main(case_tag="hi",
            deck=os.path.normpath(os.path.join(_HERE, "..", "input_06_Cattaneo_HighPa")),
            default_output="/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper/output_06_Cattaneo_HighPa",
            stem="06_Cattaneo_HighPa", zoom_default=[1.5, 4.5])
