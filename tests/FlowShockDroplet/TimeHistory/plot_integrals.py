#!/usr/bin/env python3
"""Plot conserved-quantity time histories for the dodecane shock-droplet runs.

Reads `integrals.dat`-style files and, for each data column, writes a plot of
that column's time history. Column 1 (Time) is the x-axis for the remaining
columns.

Usage:
    plot_integrals.py                     # default vap vs. no-vap overlay
    plot_integrals.py FILE                # single run, one line, no legend
    plot_integrals.py VAP_FILE NO_VAP_FILE # two runs overlaid with a legend

With a single file path, exactly one line is drawn per column and no legend is
shown. With two paths (or no arguments, which uses the two default outputs) the
runs are overlaid and a legend distinguishes them.
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
    "M_vap_transformed": r"$M_\mathrm{vap,transformed}\ [\mathrm{kg/m}]$",
    "M_floor_gas": r"$M_\mathrm{floor,gas}\ [\mathrm{kg/m}]$",
    "M_floor_liq": r"$M_\mathrm{floor,liq}\ [\mathrm{kg/m}]$",
    "E_floor": r"$E_\mathrm{floor}\ [\mathrm{J/m}]$",
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
        paths = list(DEFAULT_FILES)
    elif len(args) in (1, 2):
        paths = list(args)
    else:
        sys.exit(
            "usage: plot_integrals.py [FILE [FILE]]\n"
            "       0 args: default vap vs. no-vap overlay\n"
            "       1 arg : single run, one line, no legend\n"
            "       2 args: VAP_FILE NO_VAP_FILE overlaid with a legend"
        )

    single = len(paths) == 1

    # Build the list of series to draw. For a single file we draw one
    # unadorned line; for two we overlay the vaporizing/non-vaporizing runs.
    if single:
        series = [
            {"path": paths[0], "label": None, "linestyle": "-", "color": "C0"},
        ]
    else:
        # paths == [vap, no_vap]; draw no_vap first per PLOT_ORDER.
        by_key = {"vap": paths[0], "no_vap": paths[1]}
        series = [
            {
                "path": by_key[key],
                "label": LABELS[key],
                "linestyle": LINESTYLES[key],
                "color": COLORS[key],
            }
            for key in PLOT_ORDER
        ]

    for s in series:
        s["data"], s["names"] = load(s["path"])

    # Files may differ in column count (e.g. an older run predates a newly added
    # column like E_floor). Plot by column NAME rather than index so each series
    # contributes only the columns it actually has; the widest schema sets the
    # full list of plots, and a series missing a column is simply skipped there.
    for s in series:
        s["index"] = {name: i for i, name in enumerate(s["names"])}

    # Canonical column order: the schema of the series with the most columns.
    ref_names = max(series, key=lambda s: s["data"].shape[1])["names"]
    time_name = ref_names[0]

    # Plot every non-Time column against Time.
    for col_name in ref_names[1:]:
        fig, ax = plt.subplots(figsize=(6, 4))
        drew = False
        for s in series:
            if col_name not in s["index"] or time_name not in s["index"]:
                continue  # this run predates the column; skip it on this plot
            data = s["data"]
            ax.plot(
                data[:, s["index"][time_name]],
                data[:, s["index"][col_name]],
                label=s["label"],
                linewidth=2.0,
                linestyle=s["linestyle"],
                color=s["color"],
            )
            drew = True

        if not drew:
            plt.close(fig)
            continue

        ax.set_xlabel(LATEX_LABELS.get(time_name, time_name))
        ax.set_ylabel(LATEX_LABELS.get(col_name, col_name))
        # Force the time axis into scientific notation (shared 10^n offset).
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        if not single:
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
