import yt, numpy as np, glob, os, sys
yt.funcs.mylog.setLevel(40)

def umax(outdir):
    cells = sorted(glob.glob(os.path.join(outdir, "*cell")))
    if not cells:
        return None, None, None
    ds = yt.load(cells[-1])
    ymin = float(ds.domain_left_edge[1]); ymax = float(ds.domain_right_edge[1])
    zmid = float(0.5*(ds.domain_left_edge[2]+ds.domain_right_edge[2]))
    ray = ds.ray(ds.arr([0.0, ymin, zmid], "code_length"),
                 ds.arr([0.0, ymax, zmid], "code_length"))
    y = np.array(ray["y"]); vx = np.array(ray["velocityx"])
    return float(np.max(vx)), float(ds.current_time), len(y)

ANALYTIC = 1.25
print("="*60)
print("POISEUILLE: ENERGY BC vs PRIMITIVE (pressure) BC")
print("="*60)
print(f"  analytical u_max = {ANALYTIC:.4f} m/s\n")
for label, d in [("ENERGY BC   (UNIT_TEST_2D)", "tmp_eng"),
                 ("PRIMITIVE BC (pressure)   ", "tmp_prim")]:
    um, t, n = umax(d)
    if um is None:
        print(f"  {label}: no output in {d}/")
        continue
    print(f"  {label}: u_max = {um:.4f} m/s   (t={t:.3f}, {n} pts)   "
          f"ratio to analytic = {um/ANALYTIC:.3f}")
