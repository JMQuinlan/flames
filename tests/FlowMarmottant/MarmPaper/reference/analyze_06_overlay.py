#!/usr/bin/env python3
"""
analyze_06_overlay -- overlay R(t) of every output_06_* run in a MarmPaper output folder.

Finds every  <root>/output_06_*  directory, reads its thermo.dat (gas volume) and metadata, and draws
R(t) for all of them on one figure: one panel per drive (160 kPa / 60 kPa), each with the Cattaneo et al.
(2025) model and the Fig. 2a experiment for reference.  Extra runs (e.g. local RADIAL-slab runs) can be
added with --extra so the 3-D octant data and the slab runs can be cross-referenced on the same axes.

Radius measure (same as cattaneo_06_common.thermo_radius, the km.sim convention):
    octant / full 3-D:  R = R0 (V/V0)^(1/3)
    RADIAL slab:        R = R0  V/V0          (slab gas volume is linear in R)
V is thermo.dat column 2 (gas_volume); V0 its first value.  The octant factor 8 cancels in V/V0.
Geometry comes from the run's metadata (amr.n_cell: slab if the 2nd and 3rd entries are <= 4).
R0 and the drive amplitude come from the run's deck (input_06_* named like the output folder, or the deck
recorded in metadata) with the run's metadata overrides applied, exactly as analyze_06_Cattaneo_*.py.

Usage (on INCLINE):
    python3 analyze_06_overlay.py
    python3 analyze_06_overlay.py --root /mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper \
        --extra /path/to/slab_run:"slab goodone c19" --out overlay_06.png
Options:
    --match SUBSTR     only folders whose name contains SUBSTR (repeatable)
    --norm             plot R/R0 instead of R [um]
    --single           all runs on ONE axis (both drives) instead of one panel per drive
    --no-model / --no-exp
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import cattaneo_06_common as cc  # noqa: E402

DEFAULT_ROOT = "/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper"
DECK_DIR = os.path.dirname(_HERE)            # .../tests/FlowMarmottant/MarmPaper (the input_06_* decks)

# Okabe-Ito order (colour-blind safe); line styles cycle too so identity never rests on colour alone.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999"]
STYLES = ["-", "--", "-.", ":"]


def last_meta(plotdir, key):
    """Last assignment of `key` in <plotdir>/metadata (the value the solver used), or None."""
    f = os.path.join(plotdir, "metadata")
    if not os.path.isfile(f):
        return None
    val = None
    with open(f) as fh:
        for line in fh:
            m = re.match(r"^\s*" + re.escape(key) + r"\s*(?:\(nvals\s*=\s*\d+\)\s*::\s*\[(.*)\]|=\s*(.*))$", line.strip())
            if m:
                val = (m.group(1) or m.group(2)).strip()
    return val


def geometry(plotdir):
    nc = last_meta(plotdir, "amr.n_cell")
    if nc:
        n = [int(float(v)) for v in nc.replace(",", " ").split()[:3]]
        if len(n) == 3 and n[1] <= 4 and n[2] <= 4:
            return "slab"
    lo = last_meta(plotdir, "geometry.prob_lo")
    if lo and all(float(v) > -1e-12 for v in lo.replace(",", " ").split()[:3]):
        return "octant"
    return "full"


def find_deck(plotdir, deck_dirs):
    """input_06_* deck for an output_06_* folder (name match first, then the metadata's plot_file name)."""
    names = [os.path.basename(os.path.normpath(plotdir)).replace("output_", "input_", 1)]
    pf = last_meta(plotdir, "plot_file")
    if pf:
        names.append(os.path.basename(os.path.normpath(pf)).replace("output_", "input_", 1))
    for d in deck_dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    return None


def read_thermo(plotdir):
    """(t, V) from thermo.dat, sorted, duplicate times from restarts dropped (last one kept); tolerant of a
    partially written last line of a running job."""
    f = os.path.join(plotdir, "thermo.dat")
    if not os.path.isfile(f):
        return None
    d = np.genfromtxt(f, skip_header=1, usecols=(0, 1), invalid_raise=False)
    d = d[np.all(np.isfinite(d), axis=1)] if d.ndim == 2 else np.empty((0, 2))
    if len(d) < 2:
        return None
    _, idx = np.unique(d[::-1, 0], return_index=True)     # last occurrence of each time
    d = d[::-1][idx]
    return d[:, 0], d[:, 1]


def drive_tag(pa):
    return "hi" if pa > 100e3 else "lo"


def load_run(plotdir, label, deck_dirs, R0_fallback=None, pa_fallback=None):
    th = read_thermo(plotdir)
    if th is None:
        print("  skip %-50s (no usable thermo.dat)" % label)
        return None
    t, V = th
    geom = geometry(plotdir)
    R0, pa = R0_fallback, pa_fallback
    # The run's own metadata holds every parameter it ran with (decks may have been edited since, e.g. the
    # 60 kPa R0 2.95 -> 3.10 um change of 2026-11-11), so it comes first; the matching deck is the fallback.
    srcs = [os.path.join(plotdir, "metadata")]
    deck = find_deck(plotdir, deck_dirs)
    if deck:
        srcs.append(deck)
    for src in srcs:
        if not os.path.isfile(src):
            continue
        try:
            C = cc.load_case(src, None if src.endswith("metadata") else plotdir)
            R0, pa = C.R0, C.pa
            break
        except Exception as e:                       # noqa: BLE001  (report and try the next source)
            print("  %s: %s not usable (%s)" % (label, src, e))
    if pa is None:
        pa = 160e3 if "HighPa" in label or "_hi" in label else 60e3
    q = V / V[0]
    Rn = q if geom == "slab" else np.cbrt(q)
    print("  %-50s %-6s %s  t_end %.2f us  R0 %s" % (label, geom, drive_tag(pa), t[-1] * 1e6,
                                                    "%.3f um" % (R0 * 1e6) if R0 else "?"))
    return dict(label=label, t=t, Rn=Rn, R0=R0, pa=pa, geom=geom)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="folder holding the output_06_* runs")
    ap.add_argument("--deck-dir", action="append", default=[], help="extra folder to look for input_06_* decks")
    ap.add_argument("--extra", action="append", default=[], help="DIR[:label] extra run (e.g. a slab run)")
    ap.add_argument("--match", action="append", default=[], help="only folders containing this substring")
    ap.add_argument("--out", default=None, help="figure path (default <root>/overlay_06_R.png)")
    ap.add_argument("--norm", action="store_true", help="plot R/R0")
    ap.add_argument("--single", action="store_true", help="all runs on one axis")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-exp", action="store_true")
    ap.add_argument("--tmax", type=float, default=12.0, help="x-axis end [us]")
    a = ap.parse_args()

    deck_dirs = a.deck_dir + [a.root, DECK_DIR]
    dirs = sorted(d for d in glob.glob(os.path.join(a.root, "output_06_*")) if os.path.isdir(d))
    if a.match:
        dirs = [d for d in dirs if any(m in os.path.basename(d) for m in a.match)]
    print("%d output_06_* folders in %s" % (len(dirs), a.root))
    runs = [r for r in (load_run(d, os.path.basename(d).replace("output_06_Cattaneo_", ""), deck_dirs) for d in dirs) if r]
    for e in a.extra:
        path, _, lab = e.partition(":")
        r = load_run(path, lab or os.path.basename(os.path.normpath(path)), deck_dirs)
        if r:
            runs.append(r)
    if not runs:
        sys.exit("nothing to plot")

    tags = ["hi", "lo"] if not a.single else ["all"]
    groups = {g: [r for r in runs if g == "all" or drive_tag(r["pa"]) == g] for g in tags}
    tags = [g for g in tags if groups[g]]
    fig, axs = plt.subplots(len(tags), 1, figsize=(11, 4.2 * len(tags)), squeeze=False)
    env = cc.Envelope() if not a.no_model else None

    for ax, g in zip(axs[:, 0], tags):
        rs = groups[g]
        R0ref = next((r["R0"] for r in rs if r["R0"]), None)
        scale = (lambda r: 1.0) if a.norm else (lambda r: (r["R0"] or R0ref or 1.0) * 1e6)
        for dtag in (["hi", "lo"] if g == "all" else [g]):
            pa = 160e3 if dtag == "hi" else 60e3
            Rref = next((r["R0"] for r in rs if r["R0"] and drive_tag(r["pa"]) == dtag), R0ref)
            if env is not None and Rref:
                tc, Rc = cc.solve_cattaneo(pa, env, (a.tmax + 0.2) * 1e-6, P=dict(R0=Rref))
                ax.plot(tc * 1e6, Rc / (Rref if a.norm else 1e-6), "-.", color="k", lw=1.3,
                        label="Cattaneo et al. (2025) model, %.0f kPa" % (pa / 1e3))
            if not a.no_exp:
                te, Re = cc.load_experiment(dtag)
                k = te <= a.tmax * 1e-6
                ax.plot(te[k] * 1e6, Re[k] / ((Rref or 1.0) if a.norm else 1e-6), "o", color="#CB181D",
                        mfc="#CB181D" if dtag == "hi" else "w", ms=3.5, label="experiment, %.0f kPa" % (pa / 1e3))
        for n, r in enumerate(rs):
            ax.plot(r["t"] * 1e6, r["Rn"] * scale(r), STYLES[(n // len(COLORS)) % len(STYLES)] if n >= len(COLORS)
                    else STYLES[n % 2], color=COLORS[n % len(COLORS)], lw=1.5,
                    label="%s [%s, to %.2f us]" % (r["label"], r["geom"], r["t"][-1] * 1e6))
        title = {"hi": "160 kPa", "lo": "60 kPa", "all": "all drives"}[g]
        ax.set_title(title, loc="left")
        ax.set_xlim(-0.2, a.tmax)
        ax.set_xlabel("Time [us]")
        ax.set_ylabel("R / R0" if a.norm else "R [um]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
    out = a.out or os.path.join(a.root, "overlay_06_R.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
