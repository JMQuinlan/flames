#!/usr/bin/env python3
"""Convert the Cattaneo et al. (Nature Physics 21:590, 2025) Source Data spreadsheets into one HDF5 file.

Source (the paper's Source Data, files 41567_2025_2785_MOESM19..22_ESM.xlsx):
    MOESM19  Fig. 2a  radius R(t), and Fig. 2c centroid displacement, for p_a = 60 and 160 kPa
    MOESM20  Fig. 3g  equilibrium radius vs shape-mode degree
    MOESM21  Fig. 4a-c  per-test jetting / sonoporation table
    MOESM22  Extended Data Fig. 7  AFM apparent modulus E of the PEG hydrogels

Output: cattaneo2025_experimental.h5 next to this script (read by cattaneo_06_common.load_experiment).
Layout:
    /fig2a/hiPa/{time_us, radius_um}      /fig2a/loPa/...      (160 / 60 kPa)
    /fig2c/hiPa/{time_us, displacement_um} /fig2c/loPa/...
    /fig3g/{equilibrium_radius_um, shape_mode_degree}
    /fig4/<column>                         one dataset per spreadsheet column ('/' -> NaN, yes/no kept as text)
    /afm/<sheet>/force_map_<k>             AFM moduli [kPa] per gel and force map; /afm/mean_stats/gel_<k>
Times are on the paper's figure axis (Fig. 1 / Fig. 2a); the simulation axis is t_sim = t_fig - 1.57 us
(cattaneo_06_common.EXP_SHIFT_US).

usage: make_cattaneo2025_h5.py [SOURCE_DIR]
"""
import glob
import os
import sys

import h5py
import numpy as np
import openpyxl

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/run/media/ttryon/USBDRIVE/flames_PAPERS/___DISSERTATION/Marmottant/Cattaneo2025_ExpirimentalData"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cattaneo2025_experimental.h5")


def sheet(name, idx=0):
    f = glob.glob(os.path.join(SRC, "*%s*.xlsx" % name))[0]
    wb = openpyxl.load_workbook(f, data_only=True)
    return os.path.basename(f), (wb.worksheets[idx] if isinstance(idx, int) else wb[idx]), wb


def column(ws, col, row0):
    """Numeric column from row0 down to the first empty cell."""
    out = []
    for r in range(row0, ws.max_row + 1):
        v = ws.cell(r, col).value
        if v is None or v == "":
            break
        out.append(float(v))
    return np.array(out)


def main():
    with h5py.File(OUT, "w") as h:
        h.attrs["reference"] = "M. Cattaneo et al., Cyclic jetting enables microbubble-mediated drug delivery, " \
                               "Nature Physics 21, 590 (2025); Source Data"
        h.attrs["time_axis"] = "figure time (Fig. 1 / Fig. 2a); simulation t_sim = t_fig - 1.57 us"
        h.attrs["converted_by"] = os.path.basename(__file__)

        # ---- Fig. 2a / 2c (MOESM19): header in rows 2-4, data from row 5 ------------------------- #
        fn, ws, _ = sheet("MOESM19")
        for fig, cols, qname in (("fig2a", {"loPa": (1, 2), "hiPa": (3, 4)}, "radius_um"),
                                 ("fig2c", {"loPa": (6, 7), "hiPa": (8, 9)}, "displacement_um")):
            for case, (ct, cv) in cols.items():
                t, v = column(ws, ct, 5), column(ws, cv, 5)
                n = min(len(t), len(v))
                g = h.create_group("%s/%s" % (fig, case))
                g.create_dataset("time_us", data=t[:n]); g.create_dataset(qname, data=v[:n])
                g.attrs["p_a_kPa"] = 160.0 if case == "hiPa" else 60.0
                g.attrs["source"] = fn
        # ---- Fig. 3g (MOESM20): data from row 4 -------------------------------------------------- #
        fn, ws, _ = sheet("MOESM20")
        g = h.create_group("fig3g"); g.attrs["source"] = fn
        R, l = column(ws, 1, 4), column(ws, 2, 4); n = min(len(R), len(l))
        g.create_dataset("equilibrium_radius_um", data=R[:n]); g.create_dataset("shape_mode_degree", data=l[:n])
        # ---- Fig. 4 (MOESM21): header row 3, data from row 4 ------------------------------------- #
        fn, ws, _ = sheet("MOESM21")
        g = h.create_group("fig4"); g.attrs["source"] = fn
        names = [ws.cell(3, c).value for c in range(1, ws.max_column + 1)]
        rows = [r for r in ws.iter_rows(min_row=4, values_only=True) if r[0] is not None]
        for c, name in enumerate(names):
            if not name:
                continue
            key = name.strip().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").replace("/", "_")
            vals = [r[c] for r in rows]
            if all(isinstance(v, (int, float)) or v in ("/", None) for v in vals):
                g.create_dataset(key, data=np.array([np.nan if v in ("/", None) else float(v) for v in vals]))
            else:
                g.create_dataset(key, data=np.array([str(v).strip() for v in vals], dtype=h5py.string_dtype()))
            g[key].attrs["column"] = name
        # ---- AFM (MOESM22) ------------------------------------------------------------------------- #
        fn, _, wb = sheet("MOESM22")
        a = h.create_group("afm"); a.attrs["source"] = fn; a.attrs["units"] = "kPa (apparent compression modulus)"
        for ws in wb.worksheets:
            if ws.title.lower().startswith("gel"):
                g = a.create_group(ws.title.lower())
                for c in range(2, ws.max_column + 1):
                    v = column(ws, c, 5)
                    if len(v):
                        g.create_dataset("force_map_%d" % (c - 1), data=v)
            else:
                g = a.create_group("mean_stats")
                for c in range(2, 5):
                    g.create_dataset("gel_%d" % (c - 1), data=column(ws, c, 4))
    print("wrote", OUT)
    with h5py.File(OUT) as h:
        h.visititems(lambda k, o: print("  ", k, o.shape if isinstance(o, h5py.Dataset) else ""))


if __name__ == "__main__":
    main()
