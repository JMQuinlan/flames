"""
NACA 0012 AoA-sweep DRIVER (this script runs the solver; analysis lives in
analyze_NACA_0012.py).

  * TEMPLATE = tests/FlowAirfoil/NACA_0012/input -- the checked-in, hand-tuned
    case (Ma 0.3, viscous no-slip Re 5000, primitive BCs, single phase,
    eps = 0.01 skin).  Per case this script changes ONLY:
        plot_file                 -> <case>/out
        solid.phi.ic.bmp.filename -> <case>/phi.bmp   (rotated to the case AoA)
    Everything else runs byte-identical to the template.
  * AoA sweep: 0..20 deg in 1-deg steps (dir names aoa00..aoa20 -- the analyzer
    detects AoA from the name).
  * Scheduling: MAX_PAR=3 cases at a time, NP_RANKS=8 MPI ranks each.
  * After every finished case the polar plot is refreshed via
    analyze_NACA_0012.analyze_workdir() -- rerun that anytime standalone:
        python3 analyze_NACA_0012.py    (defaults to the shared WORK dir,
                                         bin/tests/FlowAirfoil/NACA_0012/sweep)
"""
import os, re, sys, glob, subprocess, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import analyze_NACA_0012 as m

MAX_PAR  = 3                                   # concurrent cases
NP_RANKS = 10                                   # MPI ranks per case
MPIRUN   = ["mpirun", "-np", str(NP_RANKS)]
AOAS     = list(range(0, 17, 2))               # 0..16 deg, 2-deg steps (Re 250k sweep)
WORK     = m.WORK                              # bin/tests/FlowAirfoil/NACA_0012/sweep (shared with analyzer)
TEMPLATE = os.path.join(m.TESTDIR, "input")
TIMEOUT  = 400000                              # s, per case

os.makedirs(WORK, exist_ok=True)


def plot_solid(bmp, tag):
    """Quick sanity image of the rotated solid (full + trailing-edge zoom)."""
    a = np.asarray(Image.open(bmp).convert("L")) / 255.0
    ext = [m.BMP_LO[0], m.BMP_HI[0], m.BMP_LO[1], m.BMP_HI[1]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for ax_, (xl, yl, tt) in zip(ax, [((-0.7, 0.7), (-0.3, 0.3), "airfoil"),
                                      ((0.28, 0.62), (-0.25, 0.10), "trailing-edge zoom")]):
        ax_.imshow(a, origin="upper", extent=ext, cmap="gray", aspect="equal", interpolation="nearest")
        ax_.set_xlim(*xl); ax_.set_ylim(*yl)
        ax_.set_title(f"{tag} solid phi (eps={m.EPS}): {tt}"); ax_.set_xlabel("x")
    sweep_tag = os.path.basename(os.path.normpath(WORK))
    plt.tight_layout(); plt.savefig(os.path.join(m.images_dir(sweep_tag), f"initial_solid_{tag}.png"), dpi=120); plt.close()


def prepare(aoa):
    """Case dir with a rotated bitmap + a template clone pointing at it."""
    tag = f"aoa{aoa:02d}"
    rd = os.path.join(WORK, tag); os.makedirs(rd, exist_ok=True)
    bmp = os.path.join(rd, "phi.bmp"); plot = os.path.join(rd, "out"); inp = os.path.join(rd, "input")
    m.write_phi_bmp(m.rotate(m.profile(), float(aoa)), bmp)
    txt = open(TEMPLATE).read()
    txt, n_pf = re.subn(r"(?m)^(plot_file\s*=\s*).*$", lambda mm: mm.group(1) + plot, txt)
    txt, n_bm = re.subn(r"(?m)^(solid\.phi\.ic\.bmp\.filename\s*=\s*).*$", lambda mm: mm.group(1) + bmp, txt)
    if n_pf != 1 or n_bm != 1:
        raise RuntimeError(f"template {TEMPLATE}: expected exactly one plot_file "
                           f"and one solid.phi.ic.bmp.filename line (got {n_pf}/{n_bm})")
    open(inp, "w").write(txt)
    plot_solid(bmp, tag)
    return dict(tag=tag, aoa=aoa, rd=rd, plot=plot, inp=inp)


def solve(case):
    env = dict(os.environ, OMP_NUM_THREADS="1")            # pure-MPI: 8 ranks x 1 thread
    cmd = MPIRUN + [os.path.abspath(m.SOLVER), os.path.abspath(case["inp"])]
    try:
        with open(os.path.join(case["rd"], "log"), "w") as lf:
            r = subprocess.run(cmd, cwd=case["rd"], env=env,
                               stdout=lf, stderr=subprocess.STDOUT, timeout=TIMEOUT)
        case["rc"] = r.returncode
    except Exception as e:
        case["rc"] = -1; case["err"] = repr(e)
    return case


def main():
    if not os.path.isfile(m.SOLVER):
        sys.exit(f"solver not found: {m.SOLVER} (build the 2D target: ./configure --dim=2 && make hydro2)")
    if not os.path.isfile(TEMPLATE):
        sys.exit(f"template input not found: {TEMPLATE}")

    print("=" * 60, flush=True)
    print(f"NACA 0012 AoA sweep: {AOAS[0]}..{AOAS[-1]} deg in 1-deg steps "
          f"({len(AOAS)} cases; {MAX_PAR} at a time x {NP_RANKS} MPI ranks)", flush=True)
    print(f"template: {TEMPLATE}\nwork dir: {WORK}", flush=True)
    print("=" * 60, flush=True)

    # RESUME SUPPORT: a case whose last plotfile already reached stop_time is
    # complete -- skip it (crash/restart safe).  Everything else (fresh, or
    # partially run) is re-prepared and re-run from t=0.
    stop_m = re.search(r"(?m)^stop_time\s*=\s*([0-9.eE+-]+)", open(TEMPLATE).read())
    t_stop = float(stop_m.group(1)) if stop_m else 1e30

    def is_complete(aoa):
        pfs = sorted(glob.glob(os.path.join(WORK, f"aoa{aoa:02d}", "out", "*cell")))
        if not pfs:
            return False
        try:
            import yt; yt.set_log_level(50)
            return float(yt.load(pfs[-1]).current_time) >= 0.98 * t_stop
        except Exception:
            return False

    todo = [a for a in AOAS if not is_complete(a)]
    if len(todo) < len(AOAS):
        print(f"resume: skipping {len(AOAS)-len(todo)} already-complete case(s): "
              f"{[a for a in AOAS if a not in todo]}", flush=True)

    cases = [prepare(a) for a in todo]
    print(f"prepared {len(cases)} cases (+ initial solid plots); launching ...", flush=True)

    n_done = len(AOAS) - len(todo)
    with ThreadPoolExecutor(MAX_PAR) as ex:
        futs = {ex.submit(solve, c): c for c in cases}
        for fut in as_completed(futs):
            c = fut.result()
            pfs = sorted(glob.glob(c["plot"] + "/*cell"))
            if c.get("rc") != 0 or not pfs:
                print(f"  {c['tag']}: FAILED (rc={c.get('rc')}, {c.get('err', '')})", flush=True)
                continue
            n_done += 1
            print(f"  {c['tag']}: done ({n_done}/{len(AOAS)})", flush=True)
            # Live polar refresh at a CHEAP covering-grid level (lev=3, ~40k
            # cells vs ~10.5M at finest) -- the finest-level pass on a full
            # window of snapshots peaks ~1 GB/snapshot and, stacked on top of
            # the running MPI ranks, OOM'd the machine mid-sweep once.
            # Finished cases are cached (analysis.json), so this stays O(1)
            # per completion.  Run the standalone CLI at the end for the
            # finest-level publication polar.
            try:
                m.analyze_workdir(WORK, lev=3)
            except Exception:
                print("  analysis EXCEPTION:", traceback.format_exc(), flush=True)

    print("\n==== SWEEP DONE ====", flush=True)
    print("Live polars used a coarse force grid (lev=3).  For the final,", flush=True)
    print("finest-level polar run (solvers idle, so the RAM spike is safe):", flush=True)
    print(f"  python3 {os.path.join(m.HERE, 'analyze_NACA_0012.py')} {WORK}", flush=True)


if __name__ == "__main__":
    main()
