#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
3D UNIT TEST AGGREGATOR
===============================================================================
Run after Run3DUnitTests.sh.  For every 2D unit test it:

  1. runs that test's EXISTING reference analysis script (the ones that already
     produce the hydro2-vs-analytical plots), pointed at the UNIT_TEST output;
  2. copies the plots that script produced into  results/<TestName>/  so
     everything is ORGANIZED BY TEST;
  3. best-effort parses max / average error from the script's stdout into
     results/summary.csv  (+ a printed table).  The full stdout is always
     saved to results/<TestName>/run.log so the raw numbers are never lost.

No stoplight yet (per request) -- just the plots + the error table.

LOCATION
--------
Set LOCATION below to match where the plotfiles live:
  "incline"  -> /mmfs1/home/ttryon/flames/bin/tests/...   (where the SLURM job writes)
  "desktop"  -> <repo>/bin/tests/...                       (a local copy)
The aggregator passes the resolved output dir to each analysis script as an
argument, so you do NOT have to edit the per-script toggles to switch machines.

Usage:
    python TryonSlurmFiles/3DUnitTests/aggregate_unit_tests.py
    python TryonSlurmFiles/3DUnitTests/aggregate_unit_tests.py --location desktop
    python TryonSlurmFiles/3DUnitTests/aggregate_unit_tests.py --only Couette_Single Laplace
===============================================================================
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))          # TryonSlurmFiles/3DUnitTests -> repo
RESULTS = os.path.join(_HERE, "results")

# ---------------------------------------------------------------------------
# LOCATION: where the plotfiles live (toggle here or with --location)
# ---------------------------------------------------------------------------
LOCATION = "incline"     # "incline" (/mmfs1) or "desktop" (<repo>/bin/tests)

OUTPUT_BASE = {
    "incline": "/mmfs1/home/ttryon/flames/bin/tests",
    "desktop": os.path.join(_REPO, "bin", "tests"),
}

PER_TEST_TIMEOUT_S = 900     # kill a single analysis after this long

# ---------------------------------------------------------------------------
# MANIFEST: test name -> (reference dir, analysis script, args template)
# {base} is filled with OUTPUT_BASE[LOCATION].  Args are passed on the command
# line; each analysis script accepts the output dir (and RiemannCompare also the
# case name) as argv, falling back to its own default if run standalone.
# ---------------------------------------------------------------------------
TESTS = [
    # ---- Riemann solvers (one script, case picks the .hdf5 + output dir) ----
    dict(name="Riemann_Toro1a", refdir="tests/FlowRiemannUnitTests/reference",
         script="RiemannCompare.py",
         args=["Toro1a", "{base}/FlowRiemannUnitTests/UNIT_TEST_3D/output_Toro1a", "z"]),
    dict(name="Riemann_Toro2", refdir="tests/FlowRiemannUnitTests/reference",
         script="RiemannCompare.py",
         args=["Toro2", "{base}/FlowRiemannUnitTests/UNIT_TEST_3D/output_Toro2", "z"]),
    dict(name="Riemann_Garrick", refdir="tests/FlowRiemannUnitTests/reference",
         script="RiemannCompare.py",
         args=["Garrick", "{base}/FlowRiemannUnitTests/UNIT_TEST_3D/output_Garrick", "z"]),
    # ---- Viscosity: Couette ----
    dict(name="Couette_Single", refdir="tests/FlowCouette/reference",
         script="couette_analysis.py",
         args=["{base}/FlowCouette/UNIT_TEST_3D/output_single"]),
    dict(name="Couette_Multi", refdir="tests/FlowCouette/reference",
         script="couette_analysis_multi.py",
         args=["{base}/FlowCouette/UNIT_TEST_3D/output_multi"]),
    # ---- Pressure: Poiseuille (analysis lives in the misspelled 'refrence') --
    dict(name="Poiseuille", refdir="tests/FlowPoseuille/refrence",
         script="poiseuillex_single.py",
         args=["{base}/FlowPoseuille/UNIT_TEST_3D/FlowPoiseuillex"]),
    # ---- Viscosity: Lamb-Oseen ----
    # (3D Taylor-Green skipped: no closed-form 3D TGV solution to validate against.)
    dict(name="LambOseen", refdir="tests/FlowLambOseenVortex/reference",
         script="lamb_oseen_single.py",
         args=["{base}/FlowLambOseenVortex/UNIT_TEST_3D/FlowLambOseenVortex"]),
    # ---- Embedded BC: Re40 vortex (no-slip check on the embedded cylinder) ---
    dict(name="VortexShed_Re40", refdir="tests/FlowVortexShed/reference",
         script="noslip_check.py",
         args=["{base}/FlowVortexShed/UNIT_TEST_3D/output_hydro2"]),
    # ---- Surface tension: Laplace (one aggregate script over the R x sigma matrix) ----
    dict(name="Laplace", refdir="tests/FlowLaplace/reference",
         script="analyze_Laplace_AGG_3D.py",
         args=["{base}/FlowLaplace/UNIT_TEST_3D"]),
]

# Error-line patterns (best effort; raw stdout is always kept in run.log).
_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
MAX_PATTERNS = [re.compile(p, re.I) for p in (
    rf"(?:l[\s_]*inf|linf|l_?∞|max(?:imum)?(?:\s*(?:rel(?:ative)?)?)?\s*err\w*)\D*{_NUM}",
    rf"max\D*{_NUM}",
)]
AVG_PATTERNS = [re.compile(p, re.I) for p in (
    rf"(?:l[\s_]*2|l_?2|avg|average|mean|rms)\D*err\w*\D*{_NUM}",
    rf"(?:avg|average|mean|rms|l2)\D*{_NUM}",
)]
PLOT_EXTS = (".png", ".eps", ".pdf", ".jpg", ".jpeg", ".gif")


def snapshot_plots(root):
    """{relpath: mtime} for every plot-like file under root."""
    snap = {}
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith(PLOT_EXTS):
                full = os.path.join(dp, f)
                try:
                    snap[os.path.relpath(full, root)] = os.path.getmtime(full)
                except OSError:
                    pass
    return snap


def parse_metric(text, patterns):
    """Last numeric match across the given patterns, or None."""
    val = None
    for line in text.splitlines():
        for pat in patterns:
            m = pat.search(line)
            if m:
                try:
                    val = float(m.group(1))
                except (ValueError, IndexError):
                    pass
    return val


def run_one(test, base, only):
    name = test["name"]
    if only and name not in only:
        return None
    refdir = os.path.join(_REPO, test["refdir"])
    script = test["script"]
    args = [a.format(base=base) for a in test["args"]]
    outdir_arg = next((a for a in args if "UNIT_TEST" in a), args[-1] if args else "")

    dest = os.path.join(RESULTS, name)
    os.makedirs(dest, exist_ok=True)

    print(f"\n=== {name} ===")
    print(f"    script : {test['refdir']}/{script}")
    print(f"    output : {outdir_arg}")

    if not os.path.isdir(refdir):
        print(f"    [SKIP] reference dir missing: {refdir}")
        return dict(name=name, status="NO_SCRIPT_DIR", max_err="", avg_err="", n_plots=0)
    if not os.path.isfile(os.path.join(refdir, script)):
        print(f"    [SKIP] script missing: {script}")
        return dict(name=name, status="NO_SCRIPT", max_err="", avg_err="", n_plots=0)

    # Most analyses take a single output dir; if it's clearly absent, skip the
    # run but still record it (so the summary shows what was/wasn't available).
    if outdir_arg and not outdir_arg.endswith("UNIT_TEST_3D") and not os.path.isdir(outdir_arg):
        print(f"    [SKIP] no plotfile output at {outdir_arg} (run the sim first)")
        return dict(name=name, status="NO_DATA", max_err="", avg_err="", n_plots=0)

    before = snapshot_plots(refdir)
    env = dict(os.environ, MPLBACKEND="Agg")     # headless plotting
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, script, *args], cwd=refdir, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=PER_TEST_TIMEOUT_S)
        out, rc = proc.stdout, proc.returncode
        status = "OK" if rc == 0 else f"RC={rc}"
    except subprocess.TimeoutExpired as e:
        out = (e.output or "") + f"\n[TIMEOUT after {PER_TEST_TIMEOUT_S}s]"
        status = "TIMEOUT"
    dt = time.time() - t0

    with open(os.path.join(dest, "run.log"), "w", encoding="utf-8") as f:
        f.write(out)

    # Copy newly produced / updated plots into results/<name>/, preserving subdirs.
    after = snapshot_plots(refdir)
    new_plots = [rel for rel, mt in after.items()
                 if rel not in before or mt > before[rel] + 1e-6]
    for rel in new_plots:
        src = os.path.join(refdir, rel)
        dst = os.path.join(dest, rel.replace(os.sep, "__"))
        try:
            shutil.copy2(src, dst)
        except OSError as ex:
            print(f"    [warn] could not copy {rel}: {ex}")

    max_err = parse_metric(out, MAX_PATTERNS)
    avg_err = parse_metric(out, AVG_PATTERNS)
    print(f"    status={status}  time={dt:.0f}s  plots={len(new_plots)}  "
          f"max_err={max_err}  avg_err={avg_err}")
    return dict(name=name, status=status,
                max_err="" if max_err is None else f"{max_err:.6g}",
                avg_err="" if avg_err is None else f"{avg_err:.6g}",
                n_plots=len(new_plots))


def main():
    ap = argparse.ArgumentParser(description="Aggregate 2D unit-test analyses + plots.")
    ap.add_argument("--location", choices=list(OUTPUT_BASE), default=LOCATION,
                    help="where plotfiles live (default: %(default)s)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these test names (default: all)")
    a = ap.parse_args()

    base = OUTPUT_BASE[a.location]
    only = set(a.only) if a.only else None
    os.makedirs(RESULTS, exist_ok=True)

    print("=" * 70)
    print(f"3D UNIT TEST AGGREGATOR   location={a.location}   base={base}")
    print(f"results -> {RESULTS}")
    print("=" * 70)

    rows = [r for r in (run_one(t, base, only) for t in TESTS) if r]

    csv_path = os.path.join(RESULTS, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "status", "max_err", "avg_err", "n_plots"])
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    print(f"{'TEST':<22}{'STATUS':<10}{'MAX_ERR':<14}{'AVG_ERR':<14}{'PLOTS':>6}")
    print("-" * 70)
    for r in rows:
        print(f"{r['name']:<22}{r['status']:<10}{str(r['max_err']):<14}"
              f"{str(r['avg_err']):<14}{r['n_plots']:>6}")
    print("=" * 70)
    print(f"summary  -> {csv_path}")
    print(f"plots    -> {RESULTS}/<TestName>/")
    print("(max/avg error are best-effort parsed from stdout; full numbers in each run.log)")


if __name__ == "__main__":
    main()
