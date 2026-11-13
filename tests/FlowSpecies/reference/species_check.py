#!/usr/bin/env python3
"""
Species inside the two phases, stage 1 (passive transport) -- unit checks.

  python3 species_check.py advection <run_dir> [<run_dir> ...] [--outdir DIR]   tests/FlowSpecies/input_1D_Advection
  python3 species_check.py slab      <run_dir> [--outdir DIR]                   tests/FlowSpecies/input_1D_TwoPhaseSlab
  python3 species_check.py sod       <run_dir> [<run_dir> ...] [--outdir DIR]   tests/FlowSpecies/input_1D_SodSpecies
  python3 species_check.py identical <run_dir species on> <run_dir species off> flow fields must be bit-identical

Round-off checks common to all:  sum of the species of a phase = phase partial density in every cell;  total mass of
every species constant in time (periodic) ;  no negative partial density.
advection: after one turn every profile equals its initial profile; L1 error of each species, order of convergence.
slab     : the same with a moving liquid|gas interface; uniform pressure and velocity kept.
sod      : species boundary at the contact x_c = 0.5 + u* t (exact Riemann solution), driver-gas mass right of 0.5.
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import yt
yt.funcs.mylog.setLevel(40)


def files(d): return sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))


def names(d):
    n0 = n1 = None
    for ln in open(os.path.join(d, "output", "metadata")):
        m = re.match(r"\s*species\.phase([01])\.names\s*=\s*(.*)$", ln)
        if m:
            v = m.group(2).split("#")[0].split()
            if m.group(1) == "0": n0 = v
            else: n1 = v
    return n0 or [], n1 or []


def load(pf, sp):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; L = float(ds.domain_right_edge[0]); x = (np.arange(n) + 0.5) * L / n
    q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
    return dict(t=float(ds.current_time), x=x, dx=L / n, m={s: q("species_" + s) for s in sp}, Y={s: q("Y_" + s) for s in sp},
                r0=q("rho_eta0"), r1=q("rho_eta1"), p=q("pressure"), u=q("velocityx"), rho=q("density"), eta=q("eta"))


def roundoff(d):
    n0, n1 = names(d); sp = n0 + n1; F = [load(pf, sp) for pf in files(d)]
    e0 = max(np.abs(sum(f["m"][s] for s in n0) - f["r0"]).max() / f["r0"].max() for f in F) if n0 else 0.0
    e1 = max(np.abs(sum(f["m"][s] for s in n1) - f["r1"]).max() / f["r1"].max() for f in F) if n1 else 0.0
    M = np.array([[f["m"][s].sum() * f["dx"] for s in sp] for f in F])
    drift = np.abs(M - M[0]).max(axis=0) / np.maximum(np.abs(M[0]), 1e-300); mn = min(min(f["m"][s].min() for s in sp) for f in F)
    return n0, n1, sp, F, e0, e1, drift, mn


def report_roundoff(d, n0, n1, sp, e0, e1, drift, mn, conserved=True):
    print(f"{d}: phase 0 species {n0}, phase 1 species {n1}")
    print(f"  max |sum of species - phase partial density| / max: phase 0 {e0:.1e}, phase 1 {e1:.1e};   smallest partial density {mn:.2e}")
    if conserved: print("  relative drift of the total mass of each species: " + ", ".join(f"{s} {v:.1e}" for s, v in zip(sp, drift)))


def advection(dirs, outdir):
    R = []
    for d in dirs:
        n0, n1, sp, F, e0, e1, drift, mn = roundoff(d); report_roundoff(d, n0, n1, sp, e0, e1, drift, mn)
        a, b = F[0], min(F, key=lambda f: abs(f["t"] - 1.0))
        err = {s: np.abs(b["Y"][s] - a["Y"][s]).mean() for s in sp}
        tr = [s for s in sp if a["Y"][s].max() < 1e-6]
        for s in tr:
            away = (np.abs(a["x"] - 0.5) > 0.35)        # far from the top hat of the trace
            print(f"  trace {s}: uniform part {b['Y'][s][away].min():.15e} .. {b['Y'][s][away].max():.15e} (initial {a['Y'][s][away].min():.15e});"
                  f" total mass kept to {drift[sp.index(s)]:.1e}")
        print(f"  t = {b['t']:.4f}, {len(a['x'])} cells: L1 error against the initial profile  " + ", ".join(f"{s} {err[s]:.3e}" for s in sp))
        print(f"  flow: |u - 1| max {np.abs(b['u'] - 1).max():.1e}, |p - 1| max {np.abs(b['p'] - 1).max():.1e}")
        R.append((len(a["x"]), err, a, b, sp))
    R.sort(key=lambda r: r[0])
    for (na, ea, *_), (nb, eb, *_) in zip(R[:-1], R[1:]):
        print(f"  order {na} -> {nb}: " + ", ".join(f"{s} {np.log(ea[s] / eb[s]) / np.log(nb / na):.2f}" for s in ea if ea[s] > 1e-14 and eb[s] > 1e-14))
    n, err, a, b, sp = R[-1]
    fig, ax = plt.subplots(1, 3 if len(R) > 1 else 2, figsize=(5.2 * (3 if len(R) > 1 else 2), 4.3))
    big = [s for s in sp if a["Y"][s].max() > 1e-3]; st = max(1, n // 60)
    for k, s in enumerate(big):
        ax[0].plot(a["x"], a["Y"][s], "-", color=f"C{k}", lw=1.5); ax[0].plot(b["x"][::st], b["Y"][s][::st], "o", color=f"C{k}", ms=3.5, mfc="none")
        ax[0].text(0.02, a["Y"][s][0] + 0.02, s, color=f"C{k}", fontsize=10)
    ax[0].plot([], [], "k-", label="Exact"); ax[0].plot([], [], "ko", mfc="none", label="Numerical"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_xlabel("Position [m]"); ax[0].set_ylabel("Mass Fraction in the Phase [-]"); ax[0].set_title(f"After One Turn, {n} Cells")
    tr = [s for s in sp if a["Y"][s].max() < 1e-6]
    for s in tr:
        ax[1].semilogy(a["x"], a["Y"][s], "C3-", lw=1.5, label="Exact"); ax[1].semilogy(b["x"][::st], b["Y"][s][::st], "ko", ms=3.5, mfc="none", label="Numerical")
    ax[1].set_xlabel("Position [m]"); ax[1].set_ylabel("Mass Fraction of the Trace Species [-]"); ax[1].set_title("Trace Species"); ax[1].legend(fontsize=8)
    if len(R) > 1:
        nn = np.array([r[0] for r in R], float); c = ax[2]
        for k, s in enumerate(big): c.loglog(nn, [r[1][s] for r in R], "o-", color=f"C{k}", mfc="none", label=s)
        c.loglog(nn, R[0][1][big[0]] * (nn / nn[0]) ** -1, "-", color="0.6", lw=1, label="First Order")
        c.set_xlabel("Cells"); c.set_ylabel("Mean Error of the Mass Fraction [-]"); c.set_title("Convergence"); c.legend(fontsize=8)
        c.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)); c.grid(which="minor", alpha=0.15)
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle("Passive Advection of Five Species in Two Phases"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "species_advection.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/species_advection.png")


def slab(dirs, outdir):
    d = dirs[0]; n0, n1, sp, F, e0, e1, drift, mn = roundoff(d); report_roundoff(d, n0, n1, sp, e0, e1, drift, mn)
    a, b = F[0], min(F, key=lambda f: abs(f["t"] - 1.0))
    print(f"  t = {b['t']:.4f}: flow |u - 1| max {max(np.abs(f['u'] - 1).max() for f in F):.1e}, |p - 1| max {max(np.abs(f['p'] - 1).max() for f in F):.1e}")
    for s in sp:
        ph0 = s in n0; w = (a["eta"] > 0.5) if ph0 else (a["eta"] < 0.5)
        # mass-weighted: partial densities compare where the phase is; mass fractions only inside the phase
        print(f"  {s:>4}: L1 error of the partial density after one turn / mean = {np.abs(b['m'][s] - a['m'][s]).mean() / max(np.abs(a['m'][s]).mean(), 1e-300):.3e};"
              f"  mass fraction inside its phase: initial {a['Y'][s][w].min():.6g} .. {a['Y'][s][w].max():.6g}, final {b['Y'][s][w].min():.6g} .. {b['Y'][s][w].max():.6g}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3)); st = max(1, len(a["x"]) // 70)
    for k, s in enumerate(sp):
        ax[0].semilogy(a["x"], np.maximum(a["m"][s], 1e-16), "-", color=f"C{k}", lw=1.4, label=s); ax[0].semilogy(b["x"][::st], np.maximum(b["m"][s][::st], 1e-16), "o", color=f"C{k}", ms=3.5, mfc="none")
    ax[0].set_ylim(1e-9, 30); ax[0].set_xlabel("Position [m]"); ax[0].set_ylabel(r"Species Partial Density [kg/m$^3$]"); ax[0].set_title("After One Turn (Lines: Initial, Points: Numerical)"); ax[0].legend(fontsize=8, ncol=2)
    s0 = sum(b["m"][s] for s in n0) - b["r0"]; s1 = sum(b["m"][s] for s in n1) - b["r1"]
    ax[1].plot(b["x"], s0 / b["r0"].max(), "k-", lw=1, label="Liquid Species"); ax[1].plot(b["x"], s1 / b["r1"].max(), "C3--", lw=1, label="Gas Species")
    ax[1].set_xlabel("Position [m]"); ax[1].set_ylabel("(Sum of Species - Phase Density) / Max [-]"); ax[1].set_title("Closure Residual"); ax[1].legend(fontsize=8)
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle("Species in a Liquid Slab and in the Surrounding Gas, Advected Together"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "species_slab.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/species_slab.png")


def sod(dirs, outdir):
    US, XC0 = 0.9274526200489498, 0.5; rows = []
    for d in dirs:
        n0, n1, sp, F, e0, e1, drift, mn = roundoff(d); report_roundoff(d, n0, n1, sp, e0, e1, drift, mn, conserved=False)
        b = F[-1]; xc = XC0 + US * b["t"]; s = n0[0]; Y = b["Y"][s]; x = b["x"]
        k = np.where(Y < 0.5)[0][0]; xn = x[k - 1] + (x[k] - x[k - 1]) * (Y[k - 1] - 0.5) / (Y[k - 1] - Y[k])
        # driver gas that has crossed x = 0.5: exact = mass between the diaphragm and the contact = rho*_L u* t
        g = 1.4; pst = 0.30313017805064707; rstL = 1.0 * pst ** (1 / g)
        Mx = (b["m"][s][x > 0.5]).sum() * b["dx"]; Mex = 0.5 * rstL * US * b["t"]        # eta = 0.5: half of the mixture is phase 0
        tr = [q for q in sp if b["Y"][q].max() < 1e-6]
        w = 10 * b["dx"]; width = ((Y > 0.05) & (Y < 0.95)).sum()
        print(f"  t = {b['t']:.4f}, {len(x)} cells: species boundary (Y = 0.5) at {xn:.5f}, exact contact {xc:.5f}  ({(xn - xc) / b['dx']:+.2f} cells); width 5-95 % = {width} cells")
        print(f"  driver gas of phase 0 past the diaphragm: {Mx:.6e}   exact rho*_L u* t / 2 = {Mex:.6e}   ({100 * (Mx / Mex - 1):+.3f} %)")
        for q in tr: print(f"  trace {q}: mass fraction {b['Y'][q].min():.12e} .. {b['Y'][q].max():.12e} (initial 1e-9)")
        rows.append((len(x), b, xc, s, n0))
    n, b, xc, s, n0 = rows[-1]; st = max(1, n // 80)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot([0, xc, xc, 1], [1, 1, 0, 0], "C3-", lw=1.5, label="Exact"); ax[0].plot(b["x"][::st], b["Y"][s][::st], "ko", ms=3.5, mfc="none", label="Numerical")
    ax[0].set_xlabel("Position [m]"); ax[0].set_ylabel("Mass Fraction of the Driver Gas [-]"); ax[0].set_title("Species Boundary at the Contact"); ax[0].legend()
    ax[1].plot(b["x"], b["rho"], "k-", lw=1.2, label="Density"); ax[1].plot(b["x"], b["m"][n0[0]] + b["m"][n0[1]] + b["m"][n0[2]], "C3--", lw=1.2, label="Sum of the Species of Phase 0")
    ax[1].plot(b["x"], b["m"][n0[0]], "C0:", lw=1.5, label="Driver Gas, Phase 0"); ax[1].set_xlabel("Position [m]"); ax[1].set_ylabel(r"Density [kg/m$^3$]"); ax[1].set_title("Density and Species Partial Densities"); ax[1].legend(fontsize=8)
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle("Sod Shock Tube with the Driver and Driven Gas as Species"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "species_sod.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/species_sod.png")


def identical(dirs, outdir):
    a, b = files(dirs[0])[-1], files(dirs[1])[-1]; da, db = yt.load(a), yt.load(b)
    ca = da.covering_grid(0, da.domain_left_edge, da.domain_dimensions); cb = db.covering_grid(0, db.domain_left_edge, db.domain_dimensions)
    common = sorted(set(n for _, n in da.field_list) & set(n for _, n in db.field_list)); worst = 0.0; bad = []
    for n in common:
        e = float(np.abs(np.asarray(ca[("boxlib", n)]) - np.asarray(cb[("boxlib", n)])).max())
        if e > 0: bad.append((n, e))
        worst = max(worst, e)
    print(f"{dirs[0]} against {dirs[1]} at t = {float(da.current_time):.4f}: {len(common)} common fields, largest difference {worst:.1e}"
          + ("  -> BIT-IDENTICAL flow" if worst == 0 else "  DIFFERENT: " + ", ".join(f"{n} {e:.1e}" for n, e in bad[:8])))


if __name__ == "__main__":
    mode = sys.argv[1]; args = sys.argv[2:]
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--outdir")]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args[args.index("--outdir") + 1] if "--outdir" in args else os.path.join(here, "Images", os.path.basename(os.path.normpath(dirs[-1])))
    os.makedirs(outdir, exist_ok=True)
    dict(advection=advection, slab=slab, sod=sod, identical=identical)[mode](dirs, outdir)
