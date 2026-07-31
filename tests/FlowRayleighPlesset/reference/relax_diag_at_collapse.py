#!/usr/bin/env python3
"""Extract RelaxAndReinit (stiff pressure-relaxation) convergence vs sim time.

The 6-eqn recovers the Kapila / K.div(u) collapse mechanism ENTIRELY through the
stiff pressure relaxation.  If that relaxation fails to converge (unconverged_cells>0)
near the bubble-collapse instant, the bubble under-collapses -- exactly the symptom
(R_vol misses the minimum).  This scans a Hydro2 run log, ties each relaxation call
to the latest 'TIME =' it has seen, and reports global + near-collapse convergence.

Requires relax_diag=1 in the input (it is, in Sch20_Oscillating_Neumann_Large_3D).

Usage:
    python relax_diag_at_collapse.py <logfile> [t_collapse_s]
        t_collapse_s : collapse time to window around (default 1.83e-3 = Sch20 case-1 t_c)
"""
import re, sys, math

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
LOG = sys.argv[1]
TC  = float(sys.argv[2]) if len(sys.argv) > 2 else 1.83e-3

# RelaxAndReinit lev=0 step=1234 max_iters=5/30 max_residual=1.2e-08 max|p_relaxed-p_reinit|=3.4e+02 unconverged_cells=0
RX = re.compile(
    r"RelaxAndReinit lev=(\d+)\s+step=(\d+)\s+max_iters=(\d+)/(\d+)\s+"
    r"max_residual=([0-9.eE+\-]+|nan|-nan)\s+max\|p_relaxed-p_reinit\|=([0-9.eE+\-]+|nan|-nan)\s+"
    r"unconverged_cells=(\d+)")
TM = re.compile(r"TIME\s*=\s*([0-9.eE+\-]+)")

def f(s):
    return float("nan") if s in ("nan", "-nan") else float(s)

recs = []     # (time, step, lev, iters, maxit, res, pgap, unconv)
cur_t, n_time = 0.0, 0
with open(LOG, errors="ignore") as fh:
    for line in fh:
        mt = TM.search(line)
        if mt:
            cur_t = float(mt.group(1)); n_time += 1; continue
        mr = RX.search(line)
        if mr:
            lev, step, it, mx, res, pgap, unc = mr.groups()
            recs.append((cur_t, int(step), int(lev), int(it), int(mx), f(res), f(pgap), int(unc)))

if not recs:
    print("No 'RelaxAndReinit ... unconverged_cells=' lines found.")
    print("  -> is relax_diag=1 in the input?  is this the right log?")
    sys.exit(1)

n   = len(recs)
bad = [r for r in recs if r[7] > 0]
nan = [r for r in recs if math.isnan(r[5])]
fin = [r for r in recs if not math.isnan(r[5])]
worst_res = max(fin, key=lambda r: r[5]) if fin else None

print("=" * 74)
print(f"RelaxAndReinit convergence   ({n} relax calls, {n_time} TIME stamps; t_c~{TC:.3e}s)")
print("=" * 74)
if n_time == 0:
    print("  ** no 'TIME =' lines found -> reporting by STEP only (windowing disabled).")
print(f"  calls with unconverged_cells>0 : {len(bad)} / {n}")
if bad:
    mb = max(bad, key=lambda r: r[7])
    print(f"     WORST: {mb[7]} cell(s)  t={mb[0]:.4e}s step={mb[1]}  residual={mb[5]:.2e}")
if worst_res:
    print(f"  worst max_residual : {worst_res[5]:.2e}  t={worst_res[0]:.4e}s step={worst_res[1]} (unconv={worst_res[7]})")
if nan:
    print(f"  ** {len(nan)} call(s) had NaN residual -- relaxation BLEW UP "
          f"(first t={nan[0][0]:.4e}s step={nan[0][1]})")

win = [r for r in recs if 0.6 * TC <= r[0] <= 1.6 * TC] if n_time else []
print(f"\n  --- near collapse  [{0.6*TC:.3e} .. {1.6*TC:.3e}]s : {len(win)} calls ---")
if win:
    wb = max(win, key=lambda r: r[7])
    wf = [r for r in win if not math.isnan(r[5])]
    print(f"     max unconverged_cells = {wb[7]}   (t={wb[0]:.4e}s step={wb[1]})")
    if wf:
        wr = max(wf, key=lambda r: r[5])
        print(f"     max residual          = {wr[5]:.2e}  (t={wr[0]:.4e}s step={wr[1]})")
    print(f"     max Newton iters used = {max(r[3] for r in win)}/{win[0][4]}")
elif n_time:
    print("     (no relax lines in the window -- does the log reach t_c? try a different t_collapse_s)")

print("\n  VERDICT:", end=" ")
if not bad and not nan:
    print("relaxation converges EVERYWHERE -> NOT under-relaxing.\n"
          "           The missed minimum is elsewhere (interface diffusion / AMR c-f / CFL).")
elif win and max(r[7] for r in win) > 0:
    print("relaxation STALLS near collapse -> the 6-eqn IS under-relaxing when it matters.\n"
          "           Raise max_iter / fix the bracket, or relax per-stage (Sch20).")
else:
    print("some non-convergence, but NOT concentrated at collapse -> probably not the cause.")
print("=" * 74)
