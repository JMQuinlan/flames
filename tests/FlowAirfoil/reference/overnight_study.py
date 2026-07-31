"""
OVERNIGHT master study (unattended, 3-WIDE PARALLEL, NO-SLIP, 60 s to S.S.).

NO-SLIP wall (solid.slip=0): gives the wall a viscous-like shear layer -> better
steady state + a mechanism to shed circulation (Kutta) than the slip wall.
stop_time = 60 s baseline (low-Mach flows need ~60 s to converge).  Writes a plot
every 6 s so the live monitor can refresh flow-field images.

PHASE 1 -- eps = 0.02, 0.01, 0.005 at AoA=4 -> all 3 at once.  Per case: an
initial SOLID plot at setup; refinement + live flow-field plots come from
monitor_flow.py.  Cl-vs-eps refreshed after each case (Images/eps_convergence.png).

DECISION -- pass if best Cl > PASS_CL.

PHASE 2 (if pass) -- NACA polar: AoA sweep at the recovered eps, 3 at a time.
"""
import os, sys, math, json, glob, traceback, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import analyze_NACA_0012 as m

MAX_PAR = 3
m.DOM_LO, m.DOM_HI = (-2.0, -2.5), (4.0, 2.5)
m.N_CELL = (120, 96)
m.SLIP = 0                  # NO-SLIP wall
m.STOP_TIME = 240.0          # baseline to steady state
m.PLOT_DT = 6.0            # write a plotfile every 6 s (live monitor refreshes flow plots)
m.WORK = "/tmp/naca_overnight"
BASE_DX = (m.DOM_HI[0] - m.DOM_LO[0]) / m.N_CELL[0]
os.makedirs(m.WORK, exist_ok=True)
def Cl_th(a): return 2 * math.pi / math.sqrt(1 - m.MACH ** 2) * math.radians(a)
def te_box(a, half=0.10):
    th = math.radians(a); R = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
    te = R @ np.array([0.5 * m.CHORD, 0.0]); return ((te[0]-half, te[1]-half), (te[0]+half, te[1]+half))

def plot_solid(bmp, tag, eps):
    a = np.asarray(Image.open(bmp).convert("L")) / 255.0
    ext = [m.DOM_LO[0], m.DOM_HI[0], m.DOM_LO[1], m.DOM_HI[1]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for ax_, (xl, yl, tt) in zip(ax, [((-0.7, 0.7), (-0.3, 0.3), "airfoil"), ((0.28, 0.62), (-0.14, 0.08), "trailing-edge zoom")]):
        ax_.imshow(a, origin="upper", extent=ext, cmap="gray", aspect="equal", interpolation="nearest")
        ax_.set_xlim(*xl); ax_.set_ylim(*yl); ax_.set_title(f"{tag} solid phi (eps={eps}): {tt}"); ax_.set_xlabel("x")
    plt.tight_layout(); plt.savefig(os.path.join(m.IMAGES, f"initial_solid_{tag}.png"), dpi=120); plt.close()

def prepare(tag, aoa, eps, lvl):
    m.EPS = eps; m.BRINK = 4 * m.U / eps; m.MAX_LEVEL = lvl
    m.NX_BMP = int((m.DOM_HI[0]-m.DOM_LO[0]) / (eps/2)); m.NY_BMP = int((m.DOM_HI[1]-m.DOM_LO[1]) / (eps/2))
    m.REFINE_BOX = te_box(aoa)
    rd = os.path.join(m.WORK, tag); os.makedirs(rd, exist_ok=True)
    bmp = os.path.join(rd, "phi.bmp"); plot = os.path.join(rd, "out"); inp = os.path.join(rd, "input")
    m.write_phi_bmp(m.rotate(m.naca0012(), aoa), bmp)
    m.write_input(bmp, plot, inp)
    plot_solid(bmp, tag, eps)
    return dict(tag=tag, aoa=aoa, eps=eps, lvl=lvl, rd=rd, plot=plot, inp=inp)

def solve(case):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    try:
        with open(os.path.join(case["rd"], "log"), "w") as lf:
            r = subprocess.run([os.path.abspath(m.SOLVER), os.path.abspath(case["inp"])],
                               cwd=case["rd"], env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=400000)
        case["rc"] = r.returncode
    except Exception as e:
        case["rc"] = -1; case["err"] = repr(e)
    return case

def run_pool(cases, on_done):
    done = []
    with ThreadPoolExecutor(MAX_PAR) as ex:
        futs = {ex.submit(solve, c): c for c in cases}
        for fut in as_completed(futs):
            c = fut.result()
            pfs = sorted(glob.glob(c["plot"] + "/*cell"))
            if c.get("rc") != 0 or not pfs:
                print(f"  {c['tag']}: FAILED (rc={c.get('rc')}, {c.get('err','')})", flush=True); continue
            try:
                Fx, Fy = m.forces_pressure(c["plot"])
                c["Cl"] = Fy / (m.Q_INF * m.CHORD); c["Cd"] = Fx / (m.Q_INF * m.CHORD)
                done.append(c); on_done(c, done)
            except Exception:
                print("  force-calc EXCEPTION:", traceback.format_exc(), flush=True)
    return done

# ============================================================ PHASE 1
print("=" * 60 + f"\nPHASE 1: eps convergence (AoA=4, NO-SLIP, 60s, {MAX_PAR}-wide)\n" + "=" * 60, flush=True)
AOA1 = 4.0
cases1 = [prepare(f"eps{eps:.4f}", AOA1, eps, lvl) for eps, lvl in [(0.02, 3), (0.01, 4), (0.005, 5)]]
print(f"prepared {len(cases1)} eps cases (+ initial solid plots); launching {MAX_PAR}-wide ...", flush=True)

def on_eps(c, done):
    print(f"  eps={c['eps']}: Cl={c['Cl']:+.4f}  Cd={c['Cd']:+.4f}  (theory {Cl_th(AOA1):.3f})", flush=True)
    d = sorted(done, key=lambda x: -x["eps"]); E = [x["eps"] for x in d]; CL = [x["Cl"] for x in d]; CD = [x["Cd"] for x in d]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].axhline(Cl_th(AOA1), color="r", ls="--", label=f"thin-airfoil {Cl_th(AOA1):.2f}")
    ax[0].plot(E, CL, "o-", label="CFD Cl"); ax[0].invert_xaxis()
    ax[0].set_xlabel("eps (smaller ->)"); ax[0].set_ylabel("Cl"); ax[0].set_title("Lift vs eps (no-slip, 60s S.S.)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(E, CD, "s-"); ax[1].invert_xaxis(); ax[1].set_xlabel("eps (smaller ->)"); ax[1].set_ylabel("Cd"); ax[1].set_title("Drag vs eps"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(m.IMAGES, "eps_convergence.png"), dpi=120); plt.close()
    json.dump([{k: x[k] for k in ("eps", "lvl", "Cl", "Cd")} for x in d], open(os.path.join(m.IMAGES, "eps_convergence.json"), "w"), indent=2)
    print(f"  [eps_convergence.png updated, {len(done)} pts]", flush=True)

p1 = run_pool(cases1, on_eps)
best_cl = max((x["Cl"] for x in p1), default=0.0)
PASS_CL = 0.20
print(f"\nPHASE 1 complete. best Cl = {best_cl:+.4f}  ->  {'PASS' if best_cl > PASS_CL else 'FAIL'} (threshold {PASS_CL})", flush=True)

# ============================================================ PHASE 2
if best_cl > PASS_CL and p1:
    best = max(p1, key=lambda x: x["Cl"]); eps2 = max(best["eps"], 0.01); lvl2 = min(best["lvl"], 4)
    print("\n" + "=" * 60 + f"\nPHASE 2: NACA polar at eps={eps2} (lift recovered) — AoA sweep, {MAX_PAR}-wide\n" + "=" * 60, flush=True)
    cases2 = [prepare(f"aoa{a:02d}", float(a), eps2, lvl2) for a in [0, 2, 4, 6, 8, 10]]
    def on_aoa(c, done):
        print(f"  AoA={c['aoa']}: Cl={c['Cl']:+.4f} (theory {Cl_th(c['aoa']):.3f})  Cd={c['Cd']:+.4f}", flush=True)
        d = sorted(done, key=lambda x: x["aoa"]); A = np.array([x["aoa"] for x in d]); CL = np.array([x["Cl"] for x in d]); CD = np.array([x["Cd"] for x in d])
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
        ax[0].plot(A, CL, "o-", label="CFD"); ax[0].plot(A, [Cl_th(a) for a in A], "r--", label="thin-airfoil"); ax[0].plot(A, 0.11*A, "k:", label="NACA exp ~0.11/deg")
        ax[0].set_title("Cl"); ax[0].set_xlabel("AoA"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(A, CD, "s-"); ax[1].axhline(0.0065, color="k", ls=":", label="exp Cd0"); ax[1].set_title("Cd"); ax[1].set_xlabel("AoA"); ax[1].legend(); ax[1].grid(alpha=.3)
        with np.errstate(divide="ignore", invalid="ignore"):
            ax[2].plot(A, CL/np.where(CD == 0, np.nan, CD), "o-")
        ax[2].set_title("L/D"); ax[2].set_xlabel("AoA"); ax[2].grid(alpha=.3)
        fig.suptitle(f"NACA0012 polar @ M={m.MACH} eps={eps2} NO-SLIP (60s S.S.)")
        plt.tight_layout(); plt.savefig(os.path.join(m.IMAGES, "naca0012_polars.png"), dpi=120); plt.close()
        json.dump([{k: x[k] for k in ("aoa", "Cl", "Cd")} for x in d], open(os.path.join(m.IMAGES, "naca0012_polars.json"), "w"), indent=2)
        print(f"  [naca0012_polars.png updated, {len(done)} pts]", flush=True)
    run_pool(cases2, on_aoa)
    print("\nPHASE 2 complete.", flush=True)
else:
    print("\nLift did NOT recover -> Phase 2 skipped.", flush=True)

print("\n==== OVERNIGHT STUDY DONE ====", flush=True)
