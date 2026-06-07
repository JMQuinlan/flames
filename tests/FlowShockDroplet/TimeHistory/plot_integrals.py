#!/usr/bin/env python3
"""Plot conserved-quantity time histories for the dodecane shock-droplet runs.

Reads two `integrals.dat`-style files (vaporizing vs. non-vaporizing) and, for
each data column, writes plots containing both runs' time histories overlaid.
Column 1 (Time) is the x-axis for the remaining columns.

Usage:
    plot_integrals.py [VAP_FILE] [NO_VAP_FILE]

If no arguments are supplied, the two default run outputs are used.
"""

import os
import sys
import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FILES = [
    "/mmfs1/home/jquinlan/runs/dodecane/output_air_dodecane_Ma6_5/integrals.dat",
    "/mmfs1/home/jquinlan/runs/dodecane/output_air_dodecane_Ma6_5_no_vap/integrals.dat",
]

LABELS = {
    "vap": "Vaporization",
    "no_vap": "No vaporization",
}

# Line style per run. The non-vaporizing case is dot-dashed.
LINESTYLES = {
    "vap": "-",
    "no_vap": "-.",
}

# Line color per run (swapped from the default draw-order assignment).
COLORS = {
    "vap": "C0",
    "no_vap": "C1",
}

# Plot order (drawn first to last). No-vaporization is plotted first.
PLOT_ORDER = ("no_vap", "vap")

# LaTeX-style axis labels, with units in upright (\mathrm) font. The solver is
# 2D, so the area integrals carry per-unit-depth units (mass kg/m, momentum
# kg/s, energy J/m). Keyed by the column name parsed from the data header.
LATEX_LABELS = {
    "Time": r"$t\ (\mathrm{s})$",
    "M_total": r"$M_\mathrm{total}\ [\mathrm{kg/m}]$",
    "M_phase0_gas": r"$M_\mathrm{gas}\ [\mathrm{kg/m}]$",
    "M_phase1_liq": r"$M_\mathrm{liq}\ [\mathrm{kg/m}]$",
    "Px_total": r"$P_{x,\mathrm{total}}\ [\mathrm{kg/s}]$",
    "Py_total": r"$P_{y,\mathrm{total}}\ [\mathrm{kg/s}]$",
    "E_total": r"$E_\mathrm{total}\ [\mathrm{J/m}]$",
    "KE_total": r"$\mathrm{KE}_\mathrm{total}\ [\mathrm{J/m}]$",
    "M_vapor": r"$M_\mathrm{vapor}\ [\mathrm{kg/m}]$",
    "M_carrier": r"$M_\mathrm{carrier}\ [\mathrm{kg/m}]$",
}


def read_header(path):
    """Parse column names from a header like `# 1:Time 2:M_total ...`.

    Returns a list of names, or None if no such header is present.
    """
    with open(path) as f:
        first = f.readline().strip()
    if not first.startswith("#"):
        return None
    tokens = first.lstrip("#").split()
    names = []
    for tok in tokens:
        # tokens look like "1:Time"; strip the leading "N:" index if present
        names.append(tok.split(":", 1)[1] if ":" in tok else tok)
    return names


def load(path):
    """Return (data array of shape [nrows, ncols], list of column names)."""
    names = read_header(path)
    data = np.loadtxt(path, comments="#")
    if names is None or len(names) != data.shape[1]:
        names = ["Time"] + [f"col{i}" for i in range(1, data.shape[1])]
    return data, names


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        vap_path, no_vap_path = DEFAULT_FILES
    elif len(args) == 2:
        vap_path, no_vap_path = args
    else:
        sys.exit(
            "usage: plot_integrals.py [VAP_FILE] [NO_VAP_FILE]\n"
            "       (supply both paths or neither)"
        )

    FILES = {"vap": vap_path, "no_vap": no_vap_path}
    loaded = {key: load(path) for key, path in FILES.items()}

    # Use the vaporizing run's column names as the canonical schema; verify the
    # other file matches in column count so we are comparing like with like.
    ref_names = loaded["vap"][1]
    ncols = loaded["vap"][0].shape[1]
    for key, (data, names) in loaded.items():
        if data.shape[1] != ncols:
            raise ValueError(
                f"{FILES[key]} has {data.shape[1]} columns, expected {ncols}"
            )

    # Column 0 is Time; plot every other column against it.
    for col in range(1, ncols):
        fig, ax = plt.subplots(figsize=(6, 4))
        for key in PLOT_ORDER:
            data, names = loaded[key]
            t = data[:, 0]
            y = data[:, col]
            ax.plot(
                t,
                y,
                label=LABELS[key],
                linewidth=2.0,
                linestyle=LINESTYLES[key],
                color=COLORS[key],
            )

        col_name = ref_names[col]
        ax.set_xlabel(LATEX_LABELS.get(ref_names[0], ref_names[0]))
        ax.set_ylabel(LATEX_LABELS.get(col_name, col_name))
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        for ext in ("png", "eps"):
            out = os.path.join(HERE, f"{col_name}.{ext}")
            fig.savefig(out, dpi=150)
            print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
