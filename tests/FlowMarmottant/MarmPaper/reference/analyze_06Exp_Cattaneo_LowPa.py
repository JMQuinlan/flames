#!/usr/bin/env python3
"""
MarmPaper 06Exp, Low amplitude: the Cattaneo et al. (Nature Physics 21:590, 2025) EXPERIMENTAL configuration,
a full 3-D coated bubble under a rigid substrate (top face of the domain), p_a = 53.9 kPa (C4F10 refit; paper 60 kPa)
(deck ../input_06Exp_Cattaneo_LowPa).

Outputs (<figdir>/<stem>..., .png + .eps; paper time axis, Cattaneo Fig. 2a, ending at 12 us):
  _R, _R_full     R(t): experiment, Present Model (R_vol, R(eta = 0.5)), coated KM / RPE, Cattaneo et al. model
                  (cattaneo_06_common.py; peak/trough table printed)
  _R_noKM, _R_full_noKM   the same without the coated KM / RPE curves
  _modes_time, _modes_total, _modes_rebounds/   spherical-harmonic shape modes of the eta = 0.5 surface
  Simulation time is moved left by 0.25 us (--sim-shift, default -0.25 here; 0 = unshifted): the full-sphere run
  trails the experiment by a near-constant ~0.25 us (single transducer). Experiment and models are not shifted.
  _pmap, _pmap_compact   p - p_inf against time and distance from the bubble centre, away from (-z) and parallel
                  to (+x) the substrate, bubble interface overlaid (cattaneo_06exp.py)
  GIFs (cattaneo_06_gif.py, frames to 12 us, PNG + EPS frames kept in <stem>_gif_frames/<name>/):
    _bubble3d.gif           full bubble, eta = 0.5 isosurface
    _bubble3d_cut.gif       cut open, p - p_inf on the cut face
    _bubble3d_cutshell.gif  cut open, shell only
Everything is bundled into <figdir>/<stem>_results.tar.gz at the end (MARMPAPER_NO_BUNDLE=1 skips it).
Optional: --thermo-fig (gas thermodynamics), --gif-kinds bubble,pair,3d,3dcut,3dcutshell, --no-gif, --no-pmap.

    python3 analyze_06Exp_Cattaneo_LowPa.py --output RUN_DIR --figdir FIG_DIR
    python3 analyze_06Exp_Cattaneo_LowPa.py --output RUN_DIR --figdir FIG_DIR --gif-every 2 --nproc 4
    python3 analyze_06Exp_Cattaneo_LowPa.py --output RUN_DIR --no-gif            # figures only
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import cattaneo_06_common as cc          # noqa: E402
import cattaneo_06exp as x06           # noqa: E402

if __name__ == "__main__":
    cc.main(case_tag="lo",
            deck=os.path.normpath(os.path.join(_HERE, "..", "input_06Exp_Cattaneo_LowPa")),
            default_output="/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper/output_06Exp_Cattaneo_LowPa",
            stem="06Exp_Cattaneo_LowPa", zoom_default=[3.0, 6.0],
            add_args=x06.add_args, radii=x06.radii, post=lambda *x: x06.post(*x, case_tag="lo"))
