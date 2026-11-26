#!/usr/bin/env python3
"""
MarmPaper 06Exp, High amplitude: the Cattaneo et al. (Nature Physics 21:590, 2025) EXPERIMENTAL configuration,
a full 3-D coated bubble under a rigid substrate (top face of the domain), p_a = 142.3 kPa (C4F10 refit; paper 160 kPa)
(deck ../input_06Exp_Cattaneo_HighPa).

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
    _bubble3d_side.gif      side view level with the bubble (elevation 0, 45 deg between x and y, orthographic),
                            flipped so the substrate is at the bottom: compares with the experimental side imagery
    Every GIF is also written for the eta = 0.35 level (blue): <name>_eta035.gif
  _fig1, _fig1_eta035   (cattaneo_06_fig1.py, full text width) Cattaneo Fig. 1d frames (3.5-6.0, 9.5-12.0 us) above
                  the simulation at the same times and scale: side view of the eta = 0.5 / 0.35 isosurface
  _fig1_overlay, _fig1_overlay_eta035   the Fig. 1d frames with the simulated outline in red (--no-fig1 skips)
                  Experimental frames: Cattaneo et al. (2025) CC BY 4.0 -- credit them in the caption.
Everything is bundled into <figdir>/<stem>_results.tar.gz at the end (MARMPAPER_NO_BUNDLE=1 skips it).
Optional: --thermo-fig (gas thermodynamics), --gif-kinds bubble,pair,3d,3dcut,3dcutshell, --no-gif, --no-pmap, --no-fig1.

    python3 analyze_06Exp_Cattaneo_HighPa.py --output RUN_DIR --figdir FIG_DIR
    python3 analyze_06Exp_Cattaneo_HighPa.py --output RUN_DIR --figdir FIG_DIR --gif-every 2 --nproc 4
    python3 analyze_06Exp_Cattaneo_HighPa.py --output RUN_DIR --no-gif            # figures only
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import cattaneo_06_common as cc          # noqa: E402
import cattaneo_06exp as x06           # noqa: E402

if __name__ == "__main__":
    cc.main(case_tag="hi",
            deck=os.path.normpath(os.path.join(_HERE, "..", "input_06Exp_Cattaneo_HighPa")),
            default_output="/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper/output_06Exp_Cattaneo_HighPa",
            stem="06Exp_Cattaneo_HighPa", zoom_default=[3.0, 6.0],
            add_args=x06.add_args, radii=x06.radii, post=lambda *x: x06.post(*x, case_tag="hi"))
