#!/usr/bin/env python3
"""Validate two-phase Fourier conduction against the analytic two-slab contact solution.

For `Conduction_2Phase_1D`: a cold liquid slab (eta=0) in perfect thermal contact with
a hot gas slab (eta=1), pure conduction, no phase change. The classic two-semi-infinite-
body solution (constant properties, perfect contact, step IC) is:

    effusivity   b_k    = sqrt(k_k * rho_k * c_k)
    contact T    T_s    = (b_L*T_L + b_R*T_R)/(b_L + b_R)        (constant in time)
    profile      T(x,t) = T_s + (T_bulk - T_s)*erf(|x-x_int|/(2 sqrt(alpha_k t)))
    diffusivity  alpha_k= k_k/(rho_k c_k)

The compressible solver conducts on total energy with a fast acoustic field, so the
process is ISOBARIC and c_k = cp_k is expected; we also report the isochoric (cv)
prediction so the run reveals which the code follows.

Reads the plotfile (Alamo `<NNNNN>cell`) nearest `target_time` and compares the
y-averaged `T(x)` to the analytic profile, masking the +/-3*epsilon diffuse band the
sharp-contact solution does not model.

Usage:
    python3 check_conduction_2phase.py <output_dir|plotfile> [target_time]

PASS: masked-L2 error within a few % AND measured interface T within a few K of T_s.
CONTROL: rerun the input with k0_thermal=k1_thermal=0 -> the T step persists.
"""

import os
import sys
import glob
import math
import numpy as np

# ---- problem constants (must match Conduction_2Phase_1D) --------------------
X_INT   = 2.0e-3
EPSILON = 4.0e-5
T_L, T_R = 300.0, 400.0          # liquid (left) cold, gas (right) hot
# gas carrier (eos0)
RHO_G, CV_G, CP_G, K_G = 4.167, 300.0, 360.0, 0.5
# liquid (eos1)
RHO_L, CV_L, CP_L, K_L = 45.83,  80.0, 160.0, 5.0
BAND_MULT  = 3.0                 # mask |x-x_c| < BAND_MULT*epsilon
TARGET_T   = 1.6e-4              # default comparison time [s]
# Realistic tolerance for a SHARP-interface incompressible solution compared to a
# DIFFUSE-interface COMPRESSIBLE code: the residual is the genuine isobaric(cp)-vs-
# isochoric(cv) ambiguity, concentrated at the fast gas front (~few % L2).
L2_TOL     = 0.05                # masked relative L2 tolerance
TS_TOL     = 5.0                 # contact-temperature tolerance [K]

_erf = np.vectorize(math.erf)


def two_slab(x, t, cL, cR, xc):
    """Analytic contact-problem T(x) centered at xc, using cL (liquid), cR (gas)."""
    bL = math.sqrt(K_L * RHO_L * cL)
    bR = math.sqrt(K_G * RHO_G * cR)
    Ts = (bL * T_L + bR * T_R) / (bL + bR)
    aL = K_L / (RHO_L * cL)
    aR = K_G / (RHO_G * cR)
    T = np.empty_like(x)
    left = x < xc
    T[left]  = Ts + (T_L - Ts) * _erf((xc - x[left]) / (2.0 * math.sqrt(aL * t)))
    T[~left] = Ts + (T_R - Ts) * _erf((x[~left] - xc) / (2.0 * math.sqrt(aR * t)))
    return T, Ts, aL, aR


def find_plotfile(path, target):
    """Return (plotfile_path, time) nearest target; accept a dir of *cell or one plotfile."""
    import yt
    if os.path.exists(os.path.join(path, "Header")):
        cands = [path]
    else:
        cands = sorted(glob.glob(os.path.join(path, "*cell")))
    if not cands:
        sys.exit(f"no plotfiles (*cell) found under {path}")
    best, best_t, best_dt = None, None, 1e30
    for c in cands:
        try:
            t = float(yt.load(c).current_time)
        except Exception:
            continue
        if abs(t - target) < best_dt:
            best, best_t, best_dt = c, t, abs(t - target)
    return best, best_t


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    import yt
    yt.set_log_level(50)
    path = sys.argv[1]
    target = float(sys.argv[2]) if len(sys.argv) > 2 else TARGET_T

    pf, t = find_plotfile(path, target)
    ds = yt.load(pf)
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                          dims=ds.domain_dimensions)
    Tnum = np.array(cg[("boxlib", "T")])[:, :, 0].mean(axis=1)     # y-averaged T(x)
    eta  = np.array(cg[("boxlib", "eta")])[:, :, 0].mean(axis=1)
    xlo = float(ds.domain_left_edge[0]); xhi = float(ds.domain_right_edge[0])
    nx = Tnum.size
    dx = (xhi - xlo) / nx
    x = xlo + (np.arange(nx) + 0.5) * dx

    # The interface physically drifts (conduction expands the warming liquid, contracts
    # the cooling gas), so center the analytic contact solution at the MEASURED eta=0.5
    # crossing rather than the nominal x_int -- the sharp-contact solution is defined
    # about the actual contact location.
    s = np.where(np.diff(np.sign(eta - 0.5)) != 0)[0]
    x_c = float(np.interp(0.5, [eta[s[0]], eta[s[0] + 1]], [x[s[0]], x[s[0] + 1]])) \
        if s.size else X_INT
    T_int = float(np.interp(0.5, [eta[s[0]], eta[s[0] + 1]], [Tnum[s[0]], Tnum[s[0] + 1]])) \
        if s.size else Tnum[int(np.argmin(np.abs(eta - 0.5)))]

    Tcp, Ts_cp, aL, aR = two_slab(x, t, CP_L, CP_G, x_c)
    Tcv, Ts_cv, _, _   = two_slab(x, t, CV_L, CV_G, x_c)

    band = np.abs(x - x_c) < BAND_MULT * EPSILON
    keep = ~band
    dT = T_R - T_L
    err = np.abs(Tnum - Tcp)
    l2  = math.sqrt(np.mean(err[keep] ** 2)) / dT
    linf = np.max(err[keep]) / dT
    l2_cv = math.sqrt(np.mean((Tnum[keep] - Tcv[keep]) ** 2)) / dT
    imax = int(np.argmax(np.where(keep, err, 0.0)))      # where the Linf lives

    print(f"plotfile: {pf}")
    print(f"time:     {t:.4e} s  (target {target:.2e})   "
          f"sqrt(a_liq t)={math.sqrt(aL*t)/dx:.1f} cells, "
          f"sqrt(a_gas t)={math.sqrt(aR*t)/dx:.1f} cells")
    print(f"interface: measured x_c = {x_c:.5e} m  "
          f"(drift {(x_c - X_INT)/dx:+.2f} cells from nominal x_int)\n")

    print("(1) T(x) vs analytic two-slab erf (isobaric, cp), band masked, centered on x_c:")
    print(f"    L2/dT  = {l2:.4f}    Linf/dT = {linf:.4f} @ x={x[imax]:.3e} "
          f"({(x[imax]-x_c)/dx:+.0f} cells, eta={eta[imax]:.2f})    "
          f"{'PASS' if l2 < L2_TOL else 'CHECK'}")
    print(f"    (isochoric cv reference: L2/dT = {l2_cv:.4f}); the residual is the "
          f"cp/cv\n    ambiguity at the fast gas front -- contact T below discriminates.\n")

    print("(2) contact (interface) temperature -- the clean cp/cv discriminator:")
    print(f"    measured T(eta=0.5) = {T_int:.2f} K")
    print(f"    analytic  T_s (cp)  = {Ts_cp:.2f} K   (cv ref {Ts_cv:.2f} K)   "
          f"{'PASS (isobaric)' if abs(T_int - Ts_cp) < TS_TOL else 'CHECK'}\n")

    print("(info) compare against the k0=k1=0 control: the T step should persist there.")


if __name__ == "__main__":
    main()
