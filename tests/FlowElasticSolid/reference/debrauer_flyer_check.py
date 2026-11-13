#!/usr/bin/env python3
"""
Aluminium flyer on titanium at 700 m/s, states at t = 0.2 us against the semi-analytical values of
de Brauer, Iollo & Milcent, Commun. Comput. Phys. 22 (2017) 1362, test case 1 (their Tables 2 and 3).

  python3 debrauer_flyer_check.py <run_dir> [<run_dir> ...]

Run: tests/FlowElasticSolid/input_1D_TiSpallation with the paper's constants (see the changelog entry II.59):
their neo-Hookean constant chi corresponds to a shear modulus mu = 2 chi.
Measured: the plastic plateau on both sides of the Al|Ti interface (velocity, normal stress S_xx - p, densities) and
the elastic precursor states in the titanium and in the aluminium (between the elastic and the plastic front).
"""
import sys, glob, os
import numpy as np
import yt
yt.funcs.mylog.setLevel(50)
REF = dict(u=297.771, s=-7.546, ral=2879.483, rti=4784.050, uti=43.246, pti=0.704, rtip=4554.726, ual=695.283, pal=0.0648, ralp=2713.565)


def main():
    for d in sys.argv[1:]:
        pf = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[-1]
        ds = yt.load(pf); ds.force_periodicity(); ad = ds.all_data(); y = np.asarray(ad[("index", "y")]); xa = np.asarray(ad[("index", "x")]); m = y == y.min(); o = np.argsort(xa[m])
        g = lambda q: np.asarray(ad[("boxlib", q)])[m][o]
        x = 10 * xa[m][o]; u = 1e3 * g("velocityx"); p = g("pressure"); s = g("elastic_Sxx") - p; rho = 1e3 * g("density"); sal, sti = g("species_Al"), g("species_Ti")
        n = len(x); w = n // 120; ii = int(np.where(sal / 2.712 > sti / 4.527)[0].max()); al = slice(ii - 4 * w, ii - w); ti = slice(ii + w, ii + 4 * w)
        up = u[ti].mean()
        a = np.where((x > x[ii]) & (u > 0.6 * up))[0].max(); b = np.where((x > x[ii]) & (u > 2.0))[0].max(); pt = slice(a + (b - a) // 3, b - (b - a) // 3)
        b2 = np.where((x < x[ii]) & (x > 1.2) & (u > 699.9))[0].max(); a2 = np.where((x < x[ii]) & (x > 1.2) & (u > 0.5 * (up + 700.0)))[0].max(); pa = slice(b2 + (a2 - b2) // 3, a2 - (a2 - b2) // 3)
        e = lambda v, r: "%.4g (ref %.4g, %+.2f %%)" % (v, r, 100 * (v / r - 1))
        print(f"== {d}: {n} cells, t = {10 * float(ds.current_time):.3f} us, interface at {x[ii]:.3f} mm")
        print("   plastic plateau   velocity Al side " + e(u[al].mean(), REF["u"]) + ", Ti side " + e(up, REF["u"]))
        print("                     normal stress Al side " + e(s[al].mean(), REF["s"]) + ", Ti side " + e(s[ti].mean(), REF["s"]) + " GPa")
        print("                     density Al " + e(rho[al].mean(), REF["ral"]) + ", Ti " + e(rho[ti].mean(), REF["rti"]))
        print("   Ti precursor      (plastic wave at %.2f mm, elastic front at %.2f mm, %d cells between)  velocity " % (x[a], x[b], b - a) + e(u[pt].mean(), REF["uti"]) + ", pressure " + e(p[pt].mean(), REF["pti"]) + ", density " + e(rho[pt].mean(), REF["rtip"]))
        print("   Al precursor      (elastic front at %.2f mm, plastic wave at %.2f mm, %d cells between)  velocity " % (x[b2], x[a2], a2 - b2) + e(u[pa].mean(), REF["ual"]) + ", pressure " + e(p[pa].mean(), REF["pal"]) + ", density " + e(rho[pa].mean(), REF["ralp"]))


if __name__ == "__main__":
    main()
