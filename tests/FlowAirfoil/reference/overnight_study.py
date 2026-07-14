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
        python3 analyze_NACA_0012.py /tmp/naca_overnight
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
AOAS     = list(range(0, 21))                  # 0..20 deg, 1-deg steps
WORK     = m.WORK                              # /tmp/naca_overnight (shared with analyzer)
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
    plt.tight_layout(); plt.savefig(os.path.join(m.IMAGES, f"initial_solid_{tag}.png"), dpi=120); plt.close()


def prepare(aoa):
    """Case dir with a rotated bitmap + a template clone pointing at it."""
    tag = f"aoa{aoa:02d}"
    rd = os.path.join(WORK, tag); os.makedirs(rd, exist_ok=True)
    bmp = os.path.join(rd, "phi.bmp"); plot = os.path.join(rd, "out"); inp = os.path.join(rd, "input")
    m.write_phi_bmp(m.rotate(m.naca0012(), float(aoa)), bmp)
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

    cases = [prepare(a) for a in AOAS]
    print(f"prepared {len(cases)} cases (+ initial solid plots); launching ...", flush=True)

    n_done = 0
    with ThreadPoolExecutor(MAX_PAR) as ex:
        futs = {ex.submit(solve, c): c for c in cases}
        for fut in as_completed(futs):
            c = fut.result()
            pfs = sorted(glob.glob(c["plot"] + "/*cell"))
            if c.get("rc") != 0 or not pfs:
                print(f"  {c['tag']}: FAILED (rc={c.get('rc')}, {c.get('err', '')})", flush=True)
                continue
            n_done += 1
            print(f"  {c['tag']}: done ({n_done}/{len(cases)})", flush=True)
            # live polar refresh -- analysis auto-detects every finished case
            try:
                m.analyze_workdir(WORK)
            except Exception:
                print("  analysis EXCEPTION:", traceback.format_exc(), flush=True)

    print("\n==== SWEEP DONE ====", flush=True)
    print(f"final analysis:  python3 {os.path.join(m.HERE, 'analyze_NACA_0012.py')} {WORK}", flush=True)


if __name__ == "__main__":
    main()
