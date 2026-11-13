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
    --diag             also diagnose each run's last plotfiles (--diag-frames N, default 4) or every Nth plotfile
                       (--diag-every N): gas temperature T_g = E_g/(m_g cv) and density of the trace gas (a_g < 1/2)
                       against the core, and the shell concentration Gamma.  The verdict lists, in order of first
                       appearance, cold trace gas (T < 150 K, frozen-alpha guard), dense trace gas (> 5x core),
                       hot trace gas (T > 3000 K, edge pockets) and Gamma blow-up (> 100).  --diag-only skips the plot.
        python3 analyze_06_overlay.py --diag-only --diag-every 3
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


# =========================================================================================== #
#  --diag: what killed the run?  Cold/hot trace gas and shell-concentration blow-up per frame
# =========================================================================================== #
T_COLD = 150.0      # [K] gas this far below the ~295 K liquid is the frozen-guard "cold gas" signature
T_HOT = 3000.0      # [K] overheated trace gas (no heat exchange) -> expanding edge pockets
G_BIG = 100.0       # Gamma / Gamma0 far beyond any physical compression of the shell -> option-5 blow-up


def _plotfiles(d):
    fs = [p for p in glob.glob(os.path.join(d, "*cell")) if os.path.isdir(p)]
    return sorted(fs, key=lambda s: int(re.search(r"(\d+)cell$", s).group(1)))


def diag_frame(pf, cv_g):
    """One plotfile -> dict of trace-gas temperature/density and Gamma extremes (leaf cells of all levels)."""
    import yt
    yt.set_log_level(50)
    ds = yt.load(pf)
    ad = ds.all_data()
    fld = set(n for _, n in ds.field_list)
    eta = ad["boxlib", "eta"].d
    ag = np.clip(1.0 - eta, 0.0, 1.0)
    m = ad["boxlib", "rho_eta1"].d
    E1 = ad["boxlib", "energy1"].d
    x, y, z = (ad["index", c].d for c in "xyz")
    lo = np.array([float(v) for v in ds.domain_left_edge])
    hi = np.array([float(v) for v in ds.domain_right_edge])
    c0 = np.where(lo > -1e-12, 0.0, 0.5 * (lo + hi))           # bubble centre: origin (octant/slab) or box centre
    r = np.sqrt((x - c0[0]) ** 2 + (y - c0[1]) ** 2 + (z - c0[2]) ** 2)
    band = (eta > 0.4) & (eta < 0.6)
    R5 = float(np.median(r[band])) if band.any() else np.nan
    core = ag > 0.5
    k = (ag > 1e-7) & (m > 0)                                   # every cell holding gas
    Tg = np.full_like(eta, np.nan)
    Tg[k] = E1[k] / (m[k] * cv_g)                               # ideal gas: E_g = alpha rho cv T = m cv T
    rho_g = np.full_like(eta, np.nan)
    rho_g[k] = m[k] / ag[k]
    T_core = float(np.nanmedian(Tg[core])) if core.any() else np.nan
    rho_core = float(np.nanmedian(rho_g[core])) if core.any() else np.nan
    tr = k & (ag < 0.5)                                         # liquid-side / trace gas
    out = dict(t=float(ds.current_time), R5=R5, T_core=T_core, rho_core=rho_core,
               n_cold=int(np.sum(Tg[tr] < T_COLD)), n_hot=int(np.sum(Tg[tr] > T_HOT)))
    if tr.any():
        i = np.nanargmin(np.where(tr, Tg, np.inf))
        j = np.nanargmax(np.where(tr, Tg, -np.inf))
        q = np.nanargmax(np.where(tr, rho_g, -np.inf))
        out.update(T_min=Tg[i], T_min_ag=ag[i], T_min_r=r[i] / R5, T_max=Tg[j], T_max_ag=ag[j], T_max_r=r[j] / R5,
                   rho_max_ratio=rho_g[q] / rho_core, rho_max_ag=ag[q], rho_max_r=r[q] / R5)
    G = ad["boxlib", "Gamma"].d if "Gamma" in fld else None
    if G is not None:
        near = r < 3.0 * R5 if np.isfinite(R5) else np.ones_like(r, bool)
        g = np.nanargmax(np.where(near, G, -np.inf))
        out.update(G_max=G[g], G_max_r=r[g] / R5, G_max_eta=eta[g], n_Gbig=int(np.sum(G[near] > G_BIG)),
                   G_05=float(np.nanmedian(G[band])) if band.any() else np.nan)
    return out


def verdict(rows):
    """Cause signatures in the diagnosed frames, ordered by the time each FIRST appears (the final frame of a
    dying run shows everything at once; the earliest signature is the likely cause)."""
    tests = [("COLD trace gas (T < %.0f K)" % T_COLD, lambda r: r.get("n_cold", 0) > 0),
             ("DENSE trace gas (> 5x core)", lambda r: r.get("rho_max_ratio", 0) > 5),
             ("HOT trace gas (T > %.0f K)" % T_HOT, lambda r: r.get("n_hot", 0) > 0),
             ("GAMMA blow-up (> %g)" % G_BIG, lambda r: r.get("G_max", 0) > G_BIG)]
    seen = []
    for name, f in tests:
        hit = [r for r in rows if f(r)]
        if hit:
            r = hit[0]
            where = r.get("G_max_r") if name.startswith("GAMMA") else r.get("T_min_r") if name.startswith("COLD") \
                else r.get("rho_max_r") if name.startswith("DENSE") else r.get("T_max_r")
            seen.append((r["t"], "%s first at t=%.2f us, r/R %.2f" % (name, r["t"] * 1e6, where if where is not None else np.nan)))
    if not seen:
        return "no cold/hot gas or Gamma signature in these frames"
    seen.sort(key=lambda s: s[0])
    return "; ".join(s for _, s in seen) + "   (earliest = likely cause; use --diag-every for the whole run)"


def run_diag(dirs_labels, nframes, every):
    for d, lab in dirs_labels:
        fs = _plotfiles(d)
        if not fs:
            print("\n== %s: no plotfiles" % lab)
            continue
        cv = last_meta(d, "eos1.cv")
        cv = float(cv.split()[0]) if cv else 714.286
        sel = fs[::every] if every else fs[-nframes:]
        print("\n== %s  (%d plotfiles, diagnosing %d; gas cv %.1f)" % (lab, len(fs), len(sel), cv))
        print("   t[us]   R05[um]  T_core  Tmin_trace (a_g, r/R)     Tmax_trace (a_g, r/R)     rho_g/core max (a_g, r/R)  "
              "n_cold n_hot  Gamma(.5)  Gamma_max (eta, r/R)  n_G>%g" % G_BIG)
        rows = []
        for pf in sel:
            try:
                o = diag_frame(pf, cv)
            except Exception as e:                       # noqa: BLE001
                print("   %s: %s" % (os.path.basename(pf), e))
                continue
            rows.append(o)
            print("   %6.3f  %7.3f  %6.0f  %8.1f (%.0e, %4.2f)  %8.0f (%.0e, %4.2f)  %8.2f (%.0e, %4.2f)  %6d %5d  %8.3g  %9.3g (%.4f, %4.2f)  %d" % (
                o["t"] * 1e6, o["R5"] * 1e6, o["T_core"], o.get("T_min", np.nan), o.get("T_min_ag", np.nan),
                o.get("T_min_r", np.nan), o.get("T_max", np.nan), o.get("T_max_ag", np.nan), o.get("T_max_r", np.nan),
                o.get("rho_max_ratio", np.nan), o.get("rho_max_ag", np.nan), o.get("rho_max_r", np.nan),
                o["n_cold"], o["n_hot"], o.get("G_05", np.nan), o.get("G_max", np.nan), o.get("G_max_eta", np.nan),
                o.get("G_max_r", np.nan), o.get("n_Gbig", 0)))
        if rows:
            print("   VERDICT:", verdict(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", action="store_true", help="also diagnose the last plotfiles: cold/hot trace gas, Gamma blow-up")
    ap.add_argument("--diag-frames", type=int, default=4, help="number of last plotfiles per run for --diag")
    ap.add_argument("--diag-every", type=int, default=0, help="instead: every Nth plotfile of the whole run")
    ap.add_argument("--diag-only", action="store_true", help="skip the R(t) figure")
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
    if a.diag or a.diag_only:
        dl = [(d, os.path.basename(d)) for d in dirs] + [(e.partition(":")[0], e.partition(":")[2] or e) for e in a.extra]
        run_diag(dl, a.diag_frames, a.diag_every)
        if a.diag_only:
            return
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
