#!/usr/bin/env python3
"""
Sch20 uncoated COLLAPSING bubble, 3D (Schmidmayer et al. 2020, Case 2, p_inf/p_b = 1408).

All plotting lives in tests/FlowRayleighPlesset/reference/sch20_analysis.py (class Sch20Analysis);
this file only says WHAT to analyse.  Edit the lists / dicts below, then:

    python3 analyze_Sch20_Collapsing_3D.py                      # OUTPUT_DIR below (INCLINE path)
    python3 analyze_Sch20_Collapsing_3D.py /path/to/plotfiles   # any other run of the same deck
    python3 analyze_Sch20_Collapsing_3D.py --nproc 16           # more parallel readers on a big node
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, ".")))
import sch20_analysis                                             # noqa: E402

CFG = dict(
    INPUT=os.path.normpath(os.path.join(_HERE, "..", "Sch20_Collapsing_Neumann_Large_3D")),
    OUTPUT_DIR="/mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Collapsing_Large_3D",
    CASE="Collapsing",                  # used in titles: "Collapsing" / "Oscillating" / ...
    COATED=False,                # coated: Coated KM / RPE models, elastic window, coated titles and colours
    STEM="Sch20_Collapsing_3D",                  # file names in ./Images: <STEM>_R.png, <STEM>_pressure_tR.png, ...
    N_PROC=4,                       # plotfiles read in parallel
    # ---- plots (remove a name to skip it) ------------------------------------------------- #
    PLOTS=[
        "radius",        # R(t) vs (Coated) KM / RPE
        "r_volume",      # gas-volume / shell-averaged / ray radii
        "residual",      # R(t) + percent error vs KM
        "velocity",      # wall velocity
        "probes",        # R(t) over the liquid pressure at fixed radii
        "pressure",      # pressure in the (r, t) plane
        "waves",         # outbound / inbound wave maps
        "shapes",        # 3D shapes at STYLE["shapes_times"]
        "planes",        # eta = 0.5 contours on four planes over time
        "modes",         # shape modes + spectrum at R_min
        #"gif",           # shape for every frame (GIF + PNG frames)
        # "eta_band", "ic_pressure", "driven", "csv",
    ],
    DEBUG=["thermo", "health"],   # "conservation", "thermo", "wall_balance", "reflection", "health"
    # ---- look and feel overrides (see STYLE in sch20_analysis.py for every key) ---------- #
    STYLE=dict(
        radius_measure="ray",       # "ray" (eta = 0.5 along +x; KM comparison) or "volume"
        shapes_times=(0.7, "Rmin", 1.35),
        planes_n=24,
        contours=True,              # pressure / wave map contour lines
        c_line=True, c_origin="tc", # sound-speed line from the centre at t = t_c ("t0": from R0 at t = 0)
    ),
    TITLES=dict(),                  # e.g. dict(radius="My title", planes=None)  (None = no title)
)

if __name__ == "__main__":
    sch20_analysis.main(CFG)
