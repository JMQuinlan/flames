#!/usr/bin/env python3
"""
Split-field figures of the two 2D tests of Favrie & Schmidmayer, CMAME 455 (2026) 118880, in the layout of their
Figs. 12 and 13: six times, each panel showing one field in the upper half and another in the lower half of the
(symmetric) domain.

  python3 favrie_schmidmayer_2d_plots.py impact      <run_dir> [--outdir DIR] [--times 0,0.2,0.8,2,3,4]
        input_2D_AlCylinder_PorousTi   upper: numerical schlieren, lower: mixture density          (their Fig. 12)
  python3 favrie_schmidmayer_2d_plots.py perforation <run_dir> [--outdir DIR] [--times 0,1,3,5,10,20]
        input_2D_TiCylinder_PorousAl   upper: mixture density, lower: aluminium volume fraction    (their Fig. 13)

Times in microseconds (deck time unit = 10 microseconds); the nearest plot file is used and its time is printed.
Schlieren, their eq. (122) with d = 500:
    S = 1/2 [ ln(1 + a_air |grad rho|) / max(ln(1 + a_air |grad rho|)) + 1 - exp(-d a_solid |grad rho| / max|grad rho|) ]
a_solid = eta (volume fraction of the solid phase), a_air = 1 - eta.  The aluminium volume fraction is
eta x (Y_Al/rho_ref,Al) / (Y_Al/rho_ref,Al + Y_Ti/rho_ref,Ti)  (reference-volume share of the species, as in the solver).
Also written: a CSV with the solid mass, the position of the projectile and the extreme values at every plotted time.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import yt
yt.funcs.mylog.setLevel(40)
HERE = os.path.dirname(os.path.abspath(__file__))
RHO_AL, RHO_TI = 2.7, 4.527
# dark blue - white - dark red, as the density maps of the paper
CM_RHO = LinearSegmentedColormap.from_list("fs_density", ["#03003c", "#1f4e8c", "#8fc6dc", "#f4f1ea", "#e9a37c", "#b23a26", "#7a1410"])
CM_SCH = LinearSegmentedColormap.from_list("fs_schlieren", ["#c9c6bb", "#8f8d86", "#3a3936", "#000000"])


def files(d):
    return sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))


def header_time(pf):
    with open(os.path.join(pf, "Header")) as fh:
        L = fh.read().split("\n"); nc = int(L[1]); return float(L[2 + nc + 1])


def load(pf):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, :, 0]
    nx, ny = int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1]); Lx, Ly = float(ds.domain_right_edge[0]), float(ds.domain_right_edge[1])
    rho, eta = 1e3 * q("density"), np.clip(q("eta"), 0.0, 1.0)
    vA, vT = q("Y_Al") / RHO_AL, q("Y_Ti") / RHO_TI
    aAl = eta * np.where(vA + vT > 0, vA / np.maximum(vA + vT, 1e-300), 0.0)
    gx, gy = np.gradient(rho, Lx / nx, Ly / ny); g = np.hypot(gx, gy)
    la = np.log1p((1.0 - eta) * g)
    sch = 0.5 * (la / max(la.max(), 1e-300) + 1.0 - np.exp(-500.0 * eta * g / max(g.max(), 1e-300)))
    return dict(t=10.0 * float(ds.current_time), Lx=Lx, Ly=Ly, nx=nx, ny=ny, rho=rho, eta=eta, aAl=aAl, aTi=eta - aAl, sch=sch,
                p=q("pressure"), u=q("velocityx"), m0=q("rho_eta0"), sAl=q("species_Al"), sTi=q("species_Ti"))


def figure(mode, d, outdir, times):
    pfs = files(d); tt = np.array([header_time(p) for p in pfs]) * 10.0
    upper = dict(impact=("sch", CM_SCH, (0.0, 1.0), "Numerical Schlieren [-]"), perforation=("rho", CM_RHO, None, "Mixture Density [kg/m$^3$]"))[mode]
    lower = dict(impact=("rho", CM_RHO, None, "Mixture Density [kg/m$^3$]"), perforation=("aAl", "turbo", (0.0, 1.0), "Aluminium Volume Fraction [-]"))[mode]
    rho_max = 4250.0 if mode == "impact" else 4500.0
    reached = [t for t in times if np.abs(tt - t).min() <= 0.6 * max(np.diff(tt).max() if len(tt) > 1 else 1.0, 1e-9)]
    if len(reached) < len(times):
        print(f"WARNING: {d} ends at {tt.max():.2f} microseconds; times {[t for t in times if t not in reached]} are NOT reached and are left out")
    times = reached
    D = [load(pfs[int(np.argmin(np.abs(tt - t)))]) for t in times]
    fig, ax = plt.subplots(3, 2, figsize=(9.2, 14.2)); ims = [None, None]; rows = []
    for a, r, t in zip(ax.ravel(), D, times):
        ny2 = r["ny"] // 2; ext_u = (0, 10 * r["Lx"], 10 * r["Ly"] / 2, 10 * r["Ly"]); ext_l = (0, 10 * r["Lx"], 0, 10 * r["Ly"] / 2)     # mm
        for k, (key, cm, lim, lab), sl, ext in ((0, upper, slice(ny2, None), ext_u), (1, lower, slice(0, ny2), ext_l)):
            v = lim if lim else (0.0, rho_max)
            ims[k] = a.imshow(r[key][:, sl].T, origin="lower", extent=ext, cmap=cm, vmin=v[0], vmax=v[1], interpolation="bilinear", aspect="equal")
        x = (np.arange(r["nx"]) + 0.5) * 10 * r["Lx"] / r["nx"]; y = (np.arange(r["ny"]) + 0.5) * 10 * r["Ly"] / r["ny"]
        a.set_xlim(0, 10 * r["Lx"]); a.set_ylim(0, 10 * r["Ly"])      # both halves (each imshow resets the limits to its own extent)
        a.axhline(10 * r["Ly"] / 2, color="w", lw=0.4)
        a.text(0.03, 0.965, f"{r['t']:.1f} $\\mu$s" if abs(r["t"] - round(r["t"])) > 0.04 else f"{r['t']:.0f} $\\mu$s", transform=a.transAxes, va="top",
               color="k" if mode == "impact" else "w", fontsize=12)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
        proj = r["aAl"] if mode == "impact" else r["aTi"]                       # projectile material, where it fills the cell
        wgt = np.where(proj > 0.5, proj, 0.0) * ((x[:, None] < 30.0) if mode == "perforation" else 1.0)
        if mode == "impact": wgt = wgt * (r["aAl"] > 0.5)
        xc = float((wgt * x[:, None]).sum() / max(wgt.sum(), 1e-300))
        dA = (r["Lx"] / r["nx"]) * (r["Ly"] / r["ny"])
        rows.append((t, r["t"], r["sAl"].sum() * dA, r["sTi"].sum() * dA, xc, r["rho"].max(), r["p"].min(), r["p"].max(), np.abs(r["u"]).max()))
    a0 = ax[0, 0]
    for k, (key, cm, lim, lab) in enumerate((upper, lower)):
        cax = inset_axes(a0, width="3.2%", height="34%", loc="upper right" if k == 0 else "lower right",
                         bbox_to_anchor=(-0.10, -0.05 if k == 0 else 0.07, 1, 1), bbox_transform=a0.transAxes, borderpad=0)
        cb = fig.colorbar(ims[k], cax=cax); col = "k" if (mode == "impact" and k == 0) else "w"
        cb.set_label(lab, color=col, fontsize=7.5); cb.ax.tick_params(colors=col, labelsize=7); cb.outline.set_edgecolor(col)
        cax.yaxis.set_label_position("left")
    title = dict(impact="20% Porous Titanium from Aluminium Cylinder Impact",
                 perforation="40% Porous Aluminium from Titanium Cylinder Impact")[mode]
    sub = dict(impact="Upper Half: Numerical Schlieren, Lower Half: Mixture Density", perforation="Upper Half: Mixture Density, Lower Half: Aluminium Volume Fraction")[mode]
    fig.suptitle(title + "\n" + sub + " (Numerical; Domain 24 mm $\\times$ 24 mm)", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.005, top=0.94, wspace=0.012, hspace=0.012)
    out = os.path.join(outdir, f"fs2d_{mode}.png"); fig.savefig(out, dpi=300); plt.close(fig)
    with open(os.path.join(outdir, f"fs2d_{mode}_summary.csv"), "w") as fo:
        fo.write("t_requested_us,t_us,mass_Al,mass_Ti,x_projectile_mm,rho_max_kg_m3,p_min_GPa,p_max_GPa,u_max_km_s\n")
        for r in rows: fo.write(",".join(f"{v:.6g}" for v in r) + "\n")
    print(f"{d}: {mode}")
    print("   t [us]   mass Al    mass Ti    projectile x [mm]   rho max [kg/m3]   p min .. max [GPa]   |u| max [km/s]")
    for r in rows: print(f"  {r[1]:6.2f}  {r[2]:.6f}  {r[3]:.6f}      {r[4]:7.3f}          {r[5]:7.0f}        {r[6]:+.3f} .. {r[7]:.3f}        {r[8]:.3f}")
    print(f"  wrote {out}")


def main():
    args = sys.argv[1:]; mode = args[0]; skip = {args.index(k) + 1 for k in ("--outdir", "--times") if k in args}
    d = [a for i, a in enumerate(args[1:], 1) if not a.startswith("--") and i not in skip][0]
    outdir = args[args.index("--outdir") + 1] if "--outdir" in args else os.path.join(HERE, "Images", os.path.basename(os.path.normpath(d)))
    times = [float(v) for v in args[args.index("--times") + 1].split(",")] if "--times" in args else dict(impact=[0, 0.2, 0.8, 2, 3, 4], perforation=[0, 1, 3, 5, 10, 20])[mode]
    os.makedirs(outdir, exist_ok=True); figure(mode, d, outdir, times)


if __name__ == "__main__":
    main()
