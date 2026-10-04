#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Sch20 MARMOTTANT COATED BUBBLE -- COLLAPSING CASE
===============================================================================
Compares tests/FlowMarmottant/input_Sch20-Collapsing_Marmottant against the
Marmottant-modified Rayleigh--Plesset and Keller--Miksis references in
reference/marmottant_rpe_km.py.

Writes one plot per file (.png and .eps) into Images/:
    Sch20_Collapsing_Marmottant_R_volume    R/R0 vs t, models + measured radii
    Sch20_Collapsing_Marmottant_residual    (R_sim - R_model)/R0 against the full shell
    Sch20_Collapsing_Marmottant_sigma       sigma(R(t)) with the buckling / rupture thresholds
    Sch20_Collapsing_Marmottant_damping     shell vs liquid interfacial pressures

Two measured radii are shown: the Sch20 gas-volume radius and the radially
averaged eta = 0.5 contour.  (The single-ray eta = 0.5 radius was dropped --
the radial average carries the same information with far less scatter.)

    python3 analyze_Sch20_Collapsing_Marmottant.py
    python3 analyze_Sch20_Collapsing_Marmottant.py --models-only      # no yt / no plotfiles needed
===============================================================================
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marmottant_sch20_common as M

# --------------------------------------------------------------------------- #
#  SHELL_ONLY
# --------------------------------------------------------------------------- #
# True  -> plot only the FULL SHELL model and the UNCOATED model.  The uncoated
#          curve is kept deliberately: it is the reference that shows how much
#          the coating actually changes, so "shell only" still means two curves.
# False -> also plot the elastic-only and viscous-only decompositions, which
#          separate which shell term a discrepancy belongs to.
SHELL_ONLY = True

INPUT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "input_Sch20-Collapsing_Marmottant"))
PLOTFILES = "/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/output_Sch20_Collapsing_Marmottant"
MODEL = "KM"            # reference the plots compare against: "KM" or "RPE"

# --------------------------------------------------------------------------- #
#  Sch20 Fig. 12 (pressure in the R-t plane) and Fig. 14 (bubble shapes)
#  Rendering and look-and-feel live in ../../FlowRayleighPlesset/reference/sch20_figs.py
# --------------------------------------------------------------------------- #
MAKE_FIG12 = True
MAKE_FIG14 = True
FIG14_TIMES = (0.7, "Rmin", 2.0)       # t/tau_c values and/or "Rmin"


def sch20_figures(input_path, out_hint, stem):
    if not (MAKE_FIG12 or MAKE_FIG14):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(here, "..", "..", "FlowRayleighPlesset", "reference")))
    import sch20_figs
    kv = M.parse_input(input_path)
    shell, liq, gas, meta = M.build_case(kv, input_path)
    plotdir = next((d for d in (out_hint, meta["plot_file"]) if d and os.path.isdir(d)), None)
    if plotdir is None:
        print("[sch20_figures] no plotfile directory found; skipped")
        return
    img = os.path.join(here, "Images"); os.makedirs(img, exist_ok=True)
    if MAKE_FIG12:
        sch20_figs.fig12_pressure_tR(plotdir, meta["R0"], meta["tau_c"], os.path.join(img, stem + "_fig12_pressure_tR"))
    if MAKE_FIG14:
        sch20_figs.fig14_shapes({"Hydro2": plotdir}, meta["R0"], meta["tau_c"],
                                os.path.join(img, stem + "_fig14_shapes"), FIG14_TIMES)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--output", default=PLOTFILES,
                    help="plotfile directory (default: the INCLINE path; falls "
                         "back to the input's plot_file, then the local bin/ tree)")
    ap.add_argument("--models-only", action="store_true")
    ap.add_argument("--all-variants", action="store_true",
                    help="override SHELL_ONLY and plot every decomposition")
    ap.add_argument("--model", default=MODEL, choices=("KM", "RPE"))
    ap.add_argument("--stem", default="Sch20_Collapsing_Marmottant")
    a = ap.parse_args()
    M.run_case(a.input, out_hint=a.output,
               shell_only=(SHELL_ONLY and not a.all_variants),
               models_only=a.models_only, stem=a.stem, model=a.model)
    if not a.models_only:
        sch20_figures(a.input, a.output, a.stem)


if __name__ == "__main__":
    main()
