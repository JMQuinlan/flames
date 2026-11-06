#!/usr/bin/env python3
"""Unit tests for the Boussinesq--Scriven surface stress as implemented in
Hydro2.cpp (the Omega build in Advance).

    T_s   = sigma_tot P + 2 mu_s D_s
    sigma_tot = sigma_eff + (kappa_s - mu_s) (div_s u)
    P     = I - n (x) n
    D_s   = P sym(grad u) P
    Omega = ||grad eta|| T_s

These are algebraic identities, so they are checked exactly (to round-off)
rather than by eyeball.  Run:  python3 unit_test_boussinesq_scriven.py
"""
import numpy as np

TOL = 1e-12
fails = []

def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok: fails.append(name)

def Ds_of(n, gu):
    n = np.asarray(n, float); n = n/np.linalg.norm(n)
    P = np.eye(len(n)) - np.outer(n, n)
    return P @ (0.5*(gu + gu.T)) @ P, P

def Ts_of(n, gu, sigma, kap, mu):
    """Exactly the code path: sigma_tot*P + 2 mu D_s, with gu(i,d)=d_i u_d."""
    Ds, P = Ds_of(n, gu)
    nn = np.asarray(n,float); nn = nn/np.linalg.norm(nn)
    div_s = np.trace(gu) - nn @ gu @ nn
    return (sigma + (kap - mu)*div_s)*P + 2.0*mu*Ds, Ds, P, div_s

print("== D_s structural identities (random states, 3D) ==")
rng = np.random.default_rng(7)
sym_ok = tan_ok = tr_ok = True
for _ in range(200):
    n = rng.normal(size=3); gu = rng.normal(size=(3,3))
    Ds, P = Ds_of(n, gu)
    nn = n/np.linalg.norm(n)
    sym_ok &= np.allclose(Ds, Ds.T, atol=TOL)
    tan_ok &= np.allclose(Ds @ nn, 0.0, atol=1e-11)      # purely tangential
    div_s = np.trace(gu) - nn @ gu @ nn
    tr_ok &= abs(np.trace(Ds) - div_s) < 1e-11           # tr D_s = div_s u
check("D_s is symmetric", sym_ok)
check("D_s is tangential  (D_s . n = 0)", tan_ok)
check("tr(D_s) = div_s u", tr_ok)

print("\n== spherical motion: mu_s must cancel EXACTLY (3D) ==")
# u = Rdot (R/r)^2 rhat  ->  at the interface grad u has the radial form below.
worst = 0.0
for _ in range(50):
    n = rng.normal(size=3); n /= np.linalg.norm(n)
    Rdot_over_R = rng.normal()
    # For radial flow the surface rate of strain is isotropic in the tangent
    # plane: grad_s u = (Rdot/R) P.  Build a grad u consistent with that.
    P = np.eye(3) - np.outer(n, n)
    gu = Rdot_over_R * P + np.outer(n, n)*rng.normal()   # normal part is arbitrary
    T0,_,_,_ = Ts_of(n, gu, 0.07, 3.0e-6, 0.0)
    T1,_,_,_ = Ts_of(n, gu, 0.07, 3.0e-6, 5.0e-6)        # same kappa_s, mu_s on
    worst = max(worst, np.abs(T1-T0).max())
check("T_s independent of mu_s for radial motion", worst < 1e-15,
      f"max|dT_s| = {worst:.3e}")

print("\n== 2D (rank-1 tangent space): mu_s does NOT cancel ==")
n2 = np.array([1.0, 0.0]); P2 = np.eye(2) - np.outer(n2, n2)
gu2 = 0.3*P2 + np.outer(n2,n2)*0.11
A,_,_,_ = Ts_of(n2, gu2, 0.07, 3.0e-6, 0.0)
B,_,_,_ = Ts_of(n2, gu2, 0.07, 3.0e-6, 5.0e-6)
check("2D: mu_s changes T_s (expected; tr P = 1 there)",
      np.abs(B-A).max() > 1e-12, f"max|dT_s| = {np.abs(B-A).max():.3e}")

print("\n== pure surface shear (div_s u = 0): only mu_s acts ==")
n = np.array([0.0,0.0,1.0])
gu = np.zeros((3,3)); gu[0,1] = 0.5; gu[1,0] = 0.5     # in-plane shear, no dilatation
T0,Ds0,_,dv0 = Ts_of(n, gu, 0.07, 3.0e-6, 0.0)
T1,Ds1,_,dv1 = Ts_of(n, gu, 0.07, 3.0e-6, 5.0e-6)
check("div_s u = 0 for pure shear", abs(dv1) < TOL, f"div_s u = {dv1:.3e}")
check("mu_s alone changes T_s under pure shear", np.abs(T1-T0).max() > 1e-12,
      f"max|dT_s| = {np.abs(T1-T0).max():.3e}")
check("the change equals 2 mu_s D_s exactly",
      np.allclose(T1-T0, 2*5.0e-6*Ds1, atol=1e-18),
      f"max resid = {np.abs((T1-T0) - 2*5.0e-6*Ds1).max():.3e}")

print("\n== Omega component packing matches Hydro2.cpp (3D) ==")
# code order: [xx, yy, zz, xy, xz, yz]
ge = np.array([0.3,-0.7,0.2]); gem = np.linalg.norm(ge); n = ge/gem
gu = rng.normal(size=(3,3)); sig, kap, mu = 0.07, 3.0e-6, 5.0e-6
Ts,Ds,P,_ = Ts_of(n, gu, sig, kap, mu)
se = sig + (kap-mu)*(np.trace(gu) - n@gu@n); tw = 2.0*mu*gem
om = [se*(gem-ge[0]*ge[0]/gem)+tw*Ds[0,0], se*(gem-ge[1]*ge[1]/gem)+tw*Ds[1,1],
      se*(gem-ge[2]*ge[2]/gem)+tw*Ds[2,2], se*(-ge[0]*ge[1]/gem)+tw*Ds[0,1],
      se*(-ge[0]*ge[2]/gem)+tw*Ds[0,2], se*(-ge[1]*ge[2]/gem)+tw*Ds[1,2]]
ref = gem*Ts
check("Omega == ||grad eta|| T_s, component by component",
      np.allclose(om, [ref[0,0],ref[1,1],ref[2,2],ref[0,1],ref[0,2],ref[1,2]], atol=1e-15))

print("\n" + ("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
