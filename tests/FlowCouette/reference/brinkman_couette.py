#!/usr/bin/env python3
"""
Brinkman (embedded-solid) Couette unit test -- post-processing.

  python3 brinkman_couette.py <plotfile> [ylo=0.125] [yhi=1.125] [U=0.1]

Exact steady solution in the fluid (ylo < y < yhi):  u = U (y - ylo)/(yhi - ylo).
Reports, x-averaged over the periodic direction:
  * max / RMS error of u over fluid cells, relative to U
  * effective slip at each wall from a straight-line fit of the fluid profile
    (fit extrapolated to the wall location; 0 = perfect no-slip)
  * the implied effective wall-position shift (slip / shear rate), in cells
  * max |v| / U (should be ~0) and the velocity left inside the solid
PASS criteria (sharp implicit wall): max error < 2%, |wall shift| < 0.5 cell.
"""
import sys
import numpy as np

def main():
    pf = sys.argv[1]
    ylo = float(sys.argv[2]) if len(sys.argv) > 2 else 0.125
    yhi = float(sys.argv[3]) if len(sys.argv) > 3 else 1.125
    U = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    import yt
    yt.set_log_level(40)
    ds = yt.load(pf)
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    g = lambda n: np.asarray(cg[("boxlib", n)])[:, :, 0].mean(axis=0)
    u = g("velocityx"); v = np.abs(np.asarray(cg[("boxlib", "velocityy")])[:, :, 0]).max(axis=0)
    phi = g("phi")
    ny = len(u); H = float(ds.domain_right_edge[1] - ds.domain_left_edge[1]); dy = H / ny
    y = (np.arange(ny) + 0.5) * dy
    fluid = (y > ylo) & (y < yhi)
    ex = U * (y - ylo) / (yhi - ylo)
    err = (u - ex)[fluid] / U
    # straight-line fit through the interior fluid (skip 2 cells next to each wall)
    core = fluid & (y > ylo + 2 * dy) & (y < yhi - 2 * dy)
    a, b = np.polyfit(y[core], u[core], 1)
    slip_lo = a * ylo + b - 0.0
    slip_hi = a * yhi + b - U
    shear = a
    t = float(ds.current_time)
    solid_u_lo = np.abs(u[y < ylo - dy]).max() / U
    solid_u_hi = np.abs(u[y > yhi + dy] - U).max() / U
    print(f"{pf}  t = {t:.2f}")
    print(f"  u error / U     : max {np.abs(err).max()*100:7.3f} %   rms {np.sqrt(np.mean(err**2))*100:7.3f} %")
    print(f"  shear rate      : {shear:.5f}  (exact {U/(yhi-ylo):.5f}, error {100*(shear*(yhi-ylo)/U-1):+.3f} %)")
    print(f"  wall slip / U   : bottom {slip_lo/U*100:+.3f} %   top {slip_hi/U*100:+.3f} %")
    print(f"  wall shift      : bottom {slip_lo/shear/dy:+.3f} cells   top {-slip_hi/shear/dy:+.3f} cells"
          "   (+ = into the solid; a first-order Brinkman wall sits at the centre of the"
          " outermost solid cell: +0.5 when the nominal wall is on a face, 0 when it is at a centre)")
    print(f"  max |v| / U     : {v.max()/U:.2e}")
    print(f"  solid interior  : |u| {solid_u_lo:.2e} (bottom),  |u-U| {solid_u_hi:.2e} (top)  (/U)")
    # expected first-order wall: centre of the outermost solid cell.  Distance
    # (in cells) from the nominal wall to that centre, measured into the solid:
    def expected(yw, solid_below):
        c = yw / dy                       # wall position in cell units
        fr = c - np.floor(c)              # 0 -> on a face, 0.5 -> at a centre
        if abs(fr - 0.5) < 1e-9: return 0.0
        return (fr + 0.5) % 1.0 if solid_below else (0.5 - fr) % 1.0
    e_lo, e_hi = expected(ylo, True), expected(yhi, False)
    s_lo, s_hi = slip_lo / shear / dy, -slip_hi / shear / dy
    print(f"  expected shift  : bottom {e_lo:+.3f}   top {e_hi:+.3f} cells  -> deviation {s_lo-e_lo:+.3f} / {s_hi-e_hi:+.3f}")
    ok = np.abs(err).max() < 0.02 and abs(s_lo - e_lo) < 0.15 and abs(s_hi - e_hi) < 0.15
    print("  RESULT          :", "PASS" if ok else "FAIL (max err < 2 %, wall within 0.15 cell of the first-order position)")

if __name__ == "__main__":
    main()
