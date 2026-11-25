#!/usr/bin/env python3
"""Interface-averaged evaporation flux j = int(m_dot dA) / int(|grad eta| dA), and film-model diagnostics in the
interface band (0.1 < eta < 0.9), for each plotfile of a run.  Usage: evap_flux.py RUN_DIR [RUN_DIR ...]"""
import sys, glob, os, numpy as np, yt
yt.funcs.mylog.setLevel(50)
for run in sys.argv[1:]:
    d = os.path.join(run, "output") if os.path.isdir(os.path.join(run, "output")) else run
    for p in sorted(glob.glob(os.path.join(d, "*[0-9]cell"))):
        ds = yt.load(p); ad = ds.all_data(); v = np.asarray(ad["index", "cell_volume"])
        md = np.asarray(ad["boxlib", "Vap_dot_rho"]); g = np.hypot(np.asarray(ad["boxlib", "grad_etax"]), np.asarray(ad["boxlib", "grad_etay"]))
        eta = np.asarray(ad["boxlib", "eta"]); w = (eta > 0.1) & (eta < 0.9)
        f = lambda n: np.asarray(ad["boxlib", n])
        pre = "Vap_film_" if ("boxlib", "Vap_film_T_liquid") in ds.field_list else "Vap_film"
        has = ("boxlib", pre + "T_liquid") in ds.field_list
        s = ""
        if has:
            wt = g[w] * v[w]
            av = lambda n: (f(n)[w] * wt).sum() / wt.sum()
            s = "Ts %.2f Tg %.2f BM %.5f BT %.5f QL/|grad| %.3g W/m2" % (av(pre + "T_liquid"), av(pre + "T_gas"), av(pre + "B_M"), av(pre + "B_T"),
                 (f(pre + "Q_L") * v).sum() / (g * v).sum())
        per = (g * v).sum()
        if per <= 0: continue
        print(f"{run} t={float(ds.current_time)*1e6:7.2f}us  j={(md*v).sum()/per:.4e} kg/m2/s  perimeter={per*1e3:.4f} mm  {s}")
