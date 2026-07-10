#!/usr/bin/env python3
"""Run the FlowRayleighPlesset diagnostic suite on one plotfile directory.

    python run_diagnostics.py [output_dir] [--full]

    output_dir : plotfile dir (default: diag_config.DEFAULT_OUTPUT_DIR)
    --full     : also run analyze_Sch20_Oscillating_3D.py -- the full figure set
                 (R_vol + radial-avg eta=0.5, eta-band, sphericity w/ jet-depth + K).
                 Heavier (KM/RP solves + all plots); the three quick tools below run
                 by default.

Quick tools (always run): mass conservation, core/band resolution, peak velocity.
Every tool ALSO runs standalone with the same default dir:
    python check_mass_conservation.py [output_dir]
    python resolution_at_collapse.py  [output_dir]
    python peak_velocity.py           [output_dir]
    python analyze_Sch20_Oscillating_3D.py [output_dir]
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from diag_config import DEFAULT_OUTPUT_DIR
except Exception:
    DEFAULT_OUTPUT_DIR = "."

QUICK = ["check_mass_conservation.py", "resolution_at_collapse.py", "peak_velocity.py"]
FULL  = ["analyze_Sch20_Oscillating_3D.py"]


def run(tool, d):
    print("\n" + "#" * 74, flush=True)
    print(f"# {tool}    ->  {d}", flush=True)
    print("#" * 74, flush=True)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, tool), d], cwd=HERE)
        ok = (r.returncode == 0)
    except Exception as e:
        print("  FAILED to launch:", e); ok = False
    return (tool, ok, time.time() - t0)


def main():
    argv = sys.argv[1:]
    full = "--full" in argv
    argv = [a for a in argv if a != "--full"]
    d = os.path.abspath(argv[0]) if argv else DEFAULT_OUTPUT_DIR
    if not os.path.isdir(d):
        print(f"!! plotfile dir not found:\n     {d}\n"
              f"   pass a directory, or edit diag_config.DEFAULT_OUTPUT_DIR")
        sys.exit(1)
    print(f"Diagnostic suite on: {d}", flush=True)
    print(f"  quick tools: {', '.join(QUICK)}" + ("  +  --full" if full else ""), flush=True)
    tools = QUICK + (FULL if full else [])
    results = [run(t, d) for t in tools]
    print("\n" + "=" * 74)
    print("SUITE SUMMARY")
    for tool, ok, dt in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {tool:44s} {dt:7.1f}s")
    print("=" * 74)
    if not full:
        print("Add --full for the R_vol / radial-avg / eta-band / sphericity figures.")
    print("Run any tool alone:  python <tool>.py [output_dir]")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
