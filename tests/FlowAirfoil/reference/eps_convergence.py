"""
epsilon-convergence study for NACA0012 subsonic lift (the Kutta/TE question).

Shrinks the diffuse-interface width eps with MATCHED refinement (finest dx ~ eps/3),
plus a small geometric refinement box right at the (rotated) trailing edge, and
watches whether C_l climbs toward the thin-airfoil value as the TE sharpens.
If C_l -> ~0.5 as eps -> 0, the diffuse method converges to the right lift
(refinement fixes it, no cut-cell needed).  If it stays ~0, the rounded TE is
fundamental.

Uses the SLIP wall (correct for inviscid, clean drag).  AoA = 4 deg, M = 0.5.
Writes Images/eps_convergence.png + a streamline image per eps.  Run from reference/.
"""
import os, sys, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analyze_NACA_0012 as m

# --- study config (override the analyzer module) ---
m.DOM_LO, m.DOM_HI = (-2.0, -2.5), (4.0, 2.5)     # smaller domain for cost (trend, not absolute)
m.N_CELL = (120, 96)
print("RUNTIME N_CELL=", m.N_CELL, flush=True)
m.SLIP = 1                                         # slip wall
m.STOP_TIME = 12.0
m.WORK = "/tmp/naca_epsconv"
AOA = 4.0
BASE_DX = (m.DOM_HI[0] - m.DOM_LO[0]) / m.N_CELL[0]   # 0.05
Cl_theory = 2 * math.pi / math.sqrt(1 - m.MACH ** 2) * math.radians(AOA)

# rotated trailing-edge point -> small refinement box around the corner
th = math.radians(AOA); R = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
te = R @ np.array([0.5 * m.CHORD, 0.0])
HALF = 0.10
m.REFINE_BOX = ((te[0] - HALF, te[1] - HALF), (te[0] + HALF, te[1] + HALF))

# (eps, max_level): finest dx = BASE_DX / 2^level ~ eps/3
CASES = [(0.02, 3), (0.01, 4), (0.005, 5)]

results = []
for eps, lvl in CASES:
    m.EPS = eps; m.BRINK = 4 * m.U / eps; m.MAX_LEVEL = lvl
    m.NX_BMP = int((m.DOM_HI[0] - m.DOM_LO[0]) / (eps / 2))   # BMP resolves eps/2
    m.NY_BMP = int((m.DOM_HI[1] - m.DOM_LO[1]) / (eps / 2))
    finest_dx = BASE_DX / 2 ** lvl
    print(f"\n=== eps={eps}  max_level={lvl}  finest_dx={finest_dx:.5f} (eps/dx={eps/finest_dx:.1f})  "
          f"BMP={m.NX_BMP}x{m.NY_BMP} ===", flush=True)
    r = m.run_case(AOA, tag=f"eps{eps:.4f}")
    if r is None:
        print(f"eps={eps}: FAILED", flush=True); continue
    r.update(eps=eps, finest_dx=finest_dx, level=lvl)
    results.append(r)
    print(f"eps={eps}:  Cl(press)={r['Cl_p']:+.4f}  Cd(press)={r['Cd_p']:+.4f}  (theory Cl={Cl_theory:.3f})", flush=True)
    # streamline view per eps
    os.system(f"python3 {os.path.join(m.HERE,'plot_flowfield.py')} {r['plot']} eps{eps:.4f} 2>/dev/null")
    # incremental plot + save (so partial results survive)
    E = np.array([x["eps"] for x in results]); Clp = np.array([x["Cl_p"] for x in results])
    Cdp = np.array([x["Cd_p"] for x in results])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].axhline(Cl_theory, color="r", ls="--", label=f"thin-airfoil {Cl_theory:.2f}")
    ax[0].plot(E, Clp, "o-", label="CFD (pressure)"); ax[0].invert_xaxis()
    ax[0].set_xlabel("eps (smaller ->)"); ax[0].set_ylabel("Cl"); ax[0].set_title(f"Lift convergence (AoA={AOA}, M={m.MACH})")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(E, Cdp, "s-"); ax[1].invert_xaxis()
    ax[1].set_xlabel("eps (smaller ->)"); ax[1].set_ylabel("Cd"); ax[1].set_title("Drag"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(m.IMAGES, "eps_convergence.png"), dpi=120)
    json.dump([{k: x[k] for k in ("eps", "finest_dx", "level", "Cl_p", "Cd_p", "Cl_k", "Cd_k")} for x in results],
              open(os.path.join(m.IMAGES, "eps_convergence.json"), "w"), indent=2)
    print(f"  [updated Images/eps_convergence.png with {len(results)} pts]", flush=True)

print("\nDONE. Cl vs eps:", [(x["eps"], round(x["Cl_p"], 4)) for x in results])
