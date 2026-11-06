#!/usr/bin/env python3
"""
Gamma convergence near collapse.

The shell law  D(Gamma)/Dt = -Gamma div_s(u)  integrates EXACTLY to
    Gamma(t) = (R0 / R(t))^2
for spherical motion.  So Gamma has a pointwise analytic reference at every
radius -- the error needs no fitting and no free constant.

This sweeps max_level at FIXED epsilon, so the interface gains CELLS without
becoming physically thicker, isolating cells-per-band (eps/dx) as the
convergence parameter.

Writes:
    Images/gamma_convergence.csv    tidy long-form (level, eps_dx, R_R0, ...)
    Images/gamma_convergence.npz    same data as arrays
    Images/gamma_convergence.png/.eps
"""
import glob, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "Images")
R0 = 2.0e-6
EPS = 2.0e-7
DOMAIN = 1.0e-4
NCELL = 64


def series(d):
    import yt
    yt.funcs.mylog.setLevel(50)
    rows = []
    for pf in sorted(glob.glob(d + "/*cell"),
                     key=lambda s: int(re.search(r"(\d+)cell", s).group(1))):
        try:
            ds = yt.load(pf); ad = ds.all_data()
            eta = np.array(ad["eta"], float)
            try:   vol = np.array(ad["index", "cell_volume"], float)
            except Exception: vol = np.array(ad["cell_volume"], float)
            sym = 1
            for k in range(3):
                if float(ds.domain_left_edge[k]) > -1e-12: sym *= 2
            V = float((np.clip(1 - eta, 0, 1) * vol).sum()) * sym
            R = (3 * V / (4 * math.pi)) ** (1 / 3)
            G = np.array(ad["Gamma"], float)
            # Gamma estimator: ANGLE-AVERAGED radial profiles of eta and Gamma,
            # then Gamma interpolated to the radius where <eta> = 0.5.
            #
            # A simple average over the eta in [0.2,0.8] band does NOT work here:
            # that band is only ~1 cell thick (0.7 cells at L2, 1.4 at L5), so it
            # samples one or two cells and is dominated by which cells happen to
            # land in range.  Two datasets disagreed by 6x on the Gamma error
            # purely from that.  Binning over all angles uses O(10^3) cells per
            # radial bin and is what makes the error measurable at all.
            x = np.array(ad["x"], float); y = np.array(ad["y"], float)
            z = np.array(ad["z"], float)
            r = np.sqrt(x*x + y*y + z*z)
            nb = 200; rmax = min(float(r.max()), 3.0 * R0)
            ed = np.linspace(0.0, rmax, nb + 1)
            idx = np.clip(np.digitize(r, ed) - 1, 0, nb - 1)
            w  = np.bincount(idx, weights=vol, minlength=nb)
            ee = np.bincount(idx, weights=eta * vol, minlength=nb)
            gg = np.bincount(idx, weights=G * vol, minlength=nb)
            ok = w > 0
            rc = (0.5 * (ed[:-1] + ed[1:]))[ok]
            ep = (ee / np.maximum(w, 1e-300))[ok]
            gp = (gg / np.maximum(w, 1e-300))[ok]
            Gb = np.nan; R05 = np.nan
            for q in range(len(rc) - 1):
                if (ep[q] - 0.5) * (ep[q + 1] - 0.5) < 0:
                    f = (0.5 - ep[q]) / (ep[q + 1] - ep[q])
                    Gb  = gp[q] + f * (gp[q + 1] - gp[q])
                    R05 = rc[q] + f * (rc[q + 1] - rc[q])
                    break
            if not np.isfinite(Gb) or Gb <= 0 or not np.isfinite(R05):
                continue
            # REFERENCE RADIUS.  Use the eta=0.5 contour radius, NOT the volume
            # radius R_V = (3/4pi int(1-eta)dV)^(1/3).  R_V carries a diffuse-band
            # bias that itself CHANGES WITH RESOLUTION, so using it makes the
            # "exact" value resolution-dependent and the convergence study
            # non-monotone (measured: L3 looked worse than L2 at R/R0 ~ 0.7
            # purely from this).  Gamma is sampled AT the eta=0.5 surface, so
            # the reference must be that same surface's radius.
            ex = (R0 / R05) ** 2
            rows.append((float(ds.current_time), R05 / R0, ex, Gb,
                         100.0 * (Gb - ex) / ex))
        except Exception:
            pass
    return np.array(rows)


def main():
    levels = []
    for d in sorted(glob.glob("bin/tests/FlowMarmottant/gconv_L*")):
        lev = int(re.search(r"gconv_L(\d+)", d).group(1))
        dx = DOMAIN / NCELL / (2 ** lev)
        s = series(d)
        if len(s) < 3:
            print(f"L{lev}: only {len(s)} usable frames"); continue
        levels.append((lev, dx, EPS / dx, s))

    if not levels:
        print("no data"); return 1
    os.makedirs(IMG, exist_ok=True)

    print(f"{'level':>6} {'dx':>11} {'eps/dx':>8} {'R0/dx':>8} "
          f"{'|err| @R/R0~0.7':>16} {'|err| @R/R0~0.45':>17} {'worst |err|':>12}")
    rowsout = []
    for lev, dx, epd, s in levels:
        def at(target):
            i = int(np.argmin(np.abs(s[:, 1] - target)))
            return abs(s[i, 4]), s[i, 1]
        e70, r70 = at(0.70)
        e45, r45 = at(0.45)
        worst = float(np.max(np.abs(s[:, 4])))
        print(f"{lev:>6} {dx:11.4e} {epd:8.2f} {R0/dx:8.2f} "
              f"{e70:16.2f} {e45:17.2f} {worst:12.2f}")
        for t, rr, ex, gb, er in s:
            rowsout.append((lev, dx, epd, R0 / dx, t, rr, ex, gb, er))

    arr = np.array(rowsout)
    hdr = "level,dx,eps_over_dx,R0_over_dx,t,R_over_R0,Gamma_exact,Gamma_sim,err_pct"
    np.savetxt(os.path.join(IMG, "gamma_convergence.csv"), arr,
               delimiter=",", header=hdr, comments="", fmt="%.8g")
    np.savez_compressed(os.path.join(IMG, "gamma_convergence.npz"),
                        data=arr, columns=hdr.split(","))
    print(f"\n  wrote {IMG}/gamma_convergence.csv and .npz  ({len(arr)} rows)")

    # --- where does it reach 98% accuracy (|err| <= 2%)? ---
    # 98%-accuracy boundary.  Walk INWARD in R/R0 and report the radius at
    # which |err| crosses 2% and does NOT recover.  Taking min(R/R0 where
    # |err|<=2) instead is wrong: the signed error crosses zero on the way out,
    # so isolated deep frames dip under 2% and the metric reports them.
    print("\n  98% accuracy (|err| <= 2%) holds inward to R/R0 =")
    for lev, dx, epd, s in levels:
        o = s[np.argsort(-s[:, 1])]          # decreasing R/R0 = inward in time
        bnd = o[0, 1]
        for rr, er in zip(o[:, 1], np.abs(o[:, 4])):
            if er > 2.0:
                break
            bnd = rr
        print(f"    L{lev}  eps/dx={epd:5.2f}  R0/dx={R0/dx:6.2f}:  R/R0 >= {bnd:.3f}"
              f"   (deepest frame reached R/R0 = {s[:,1].min():.3f})")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for lev, dx, epd, s in levels:
            lab = f"L{lev}  $\\epsilon/\\Delta x$={epd:.2f}"
            ax[0].plot(s[:, 1], s[:, 4], "-o", ms=3, label=lab)
            ax[1].semilogy(s[:, 1], np.abs(s[:, 4]) + 1e-3, "-o", ms=3, label=lab)
        for a in ax:
            a.set_xlabel(r"$R/R_0$"); a.grid(alpha=.3); a.legend(fontsize=8)
            a.invert_xaxis()
        ax[0].axhspan(-2, 2, color="0.8", alpha=.5, zorder=0)
        ax[0].set_ylabel(r"$\Gamma$ error [%]"); ax[0].set_title("Signed error")
        ax[1].axhline(2.0, color="r", ls="--", lw=1)
        ax[1].set_ylabel(r"$|\Gamma$ error$|$ [%]"); ax[1].set_title("Magnitude (2% = 98% accurate)")
        fig.suptitle(r"$\Gamma$ convergence near collapse (exact: $\Gamma=(R_0/R)^2$)",
                     fontweight="bold")
        fig.tight_layout()
        for e in ("png", "eps"):
            fig.savefig(os.path.join(IMG, f"gamma_convergence.{e}"), dpi=170,
                        bbox_inches="tight")
        print(f"  wrote {IMG}/gamma_convergence.png / .eps")
    except Exception as exc:
        print(f"  [plot skipped: {exc}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
