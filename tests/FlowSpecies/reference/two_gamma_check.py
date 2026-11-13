#!/usr/bin/env python3
"""
Species-dependent equation of state inside a phase (species.eos = mixture, stage 2a) -- tests/FlowSpecies.

  python3 two_gamma_check.py tube    <run_dir> [<run_dir> ...] [--phases <run_dir> [...]] [--outdir DIR]
  python3 two_gamma_check.py contact <run_dir> [<run_dir> ...] [--outdir DIR]

tube    : input_1D_TwoGammaShockTube.  Exact two-gas Riemann solution (gamma 1.4 left, 5/3 right): left rarefaction,
          contact (the species boundary), right shock.  L1 errors of density, velocity and pressure, convergence with
          resolution, contact position from the species field.  --phases: the same problem with the two gases as the
          two PHASES (input_1D_TwoGammaShockTube_Phases), same errors, as a cross-check.
contact : input_1D_IsobaricSpeciesContact.  Exact: p = 1 and u = 1 for all time, the profile returns after one pass.
          Reports max |p - 1| and max |u - 1| over all plot files and the L1 density error at the end.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.optimize import brentq
import yt
yt.funcs.mylog.setLevel(40)
GL, GR = 1.4, 5.0 / 3.0
WL, WR = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)        # rho, u, p
X0 = 0.5


def exact(xi):
    """Two-gas Riemann solution at xi = (x - X0)/t: rho, u, p, gamma."""
    (rl, ul, pl), (rr, ur, pr) = WL, WR
    cl, cr = np.sqrt(GL * pl / rl), np.sqrt(GR * pr / rr)

    def f(p, r, pk, c, g):
        if p <= pk: return 2 * c / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)
        return (p - pk) * np.sqrt(2 / ((g + 1) * r) / (p + (g - 1) / (g + 1) * pk))
    ps = brentq(lambda p: f(p, rl, pl, cl, GL) + f(p, rr, pr, cr, GR) + ur - ul, 1e-8, 10 * max(pl, pr), xtol=1e-14)
    us = 0.5 * (ul + ur) + 0.5 * (f(ps, rr, pr, cr, GR) - f(ps, rl, pl, cl, GL))
    rsl = rl * (ps / pl) ** (1 / GL); csl = cl * (ps / pl) ** ((GL - 1) / (2 * GL))          # left rarefaction (ps < pl)
    rsr = rr * ((ps / pr + (GR - 1) / (GR + 1)) / ((GR - 1) / (GR + 1) * ps / pr + 1))      # right shock (ps > pr)
    S = ur + cr * np.sqrt((GR + 1) / (2 * GR) * ps / pr + (GR - 1) / (2 * GR))
    xi = np.asarray(xi, float); rho = np.empty_like(xi); u = np.empty_like(xi); p = np.empty_like(xi)
    head, tail = ul - cl, us - csl
    m = xi <= head; rho[m], u[m], p[m] = rl, ul, pl
    m = (xi > head) & (xi < tail)
    c = (2 / (GL + 1)) * (cl + (GL - 1) / 2 * (ul - xi[m])); u[m] = (2 / (GL + 1)) * (cl + (GL - 1) / 2 * ul + xi[m])
    rho[m] = rl * (c / cl) ** (2 / (GL - 1)); p[m] = pl * (c / cl) ** (2 * GL / (GL - 1))
    m = (xi >= tail) & (xi < us); rho[m], u[m], p[m] = rsl, us, ps
    m = (xi >= us) & (xi < S); rho[m], u[m], p[m] = rsr, us, ps
    m = xi >= S; rho[m], u[m], p[m] = rr, ur, pr
    return rho, u, p, dict(ps=ps, us=us, S=S, rsl=rsl, rsr=rsr, head=head, tail=tail)


def files(d):
    return sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))


def load(pf, names):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = int(ds.domain_dimensions[0]); L = float(ds.domain_right_edge[0]); x = (np.arange(n) + 0.5) * L / n
    have = [f[1] for f in ds.field_list]
    out = {nm: (np.asarray(cg[("boxlib", nm)])[:, 0, 0] if nm in have else None) for nm in names}
    return float(ds.current_time), x, out


def tube_one(d):
    t, x, q = load(files(d)[-1], ["density", "velocityx", "pressure", "species_A", "species_A1", "rho_eta0"])
    re, ue, pe, info = exact((x - X0) / t)
    dx = x[1] - x[0]
    # contact position from the MASS of gas A right of x = 0.4 (inside the left star state at t = 0.2 only if the
    # rarefaction tail has passed; it has not, so the exact mass is integrated from the exact density)
    mA = (q["species_A"] + q["species_A1"]) if q["species_A"] is not None else q["rho_eta0"]   # species run / phase run
    xs = X0 + info["us"] * t; m = x > 0.4
    xf = np.linspace(0.4, xs, 20001); mex = np.trapz(exact((xf - X0) / t)[0], xf)
    xc = xs + (np.sum(mA[m]) * dx - mex) / info["rsl"]              # excess mass of A / density of A at the contact
    return dict(d=d, n=len(x), t=t, x=x, rho=q["density"], u=q["velocityx"], p=q["pressure"], re=re, ue=ue, pe=pe, info=info,
                er=np.abs(q["density"] - re).sum() * dx, eu=np.abs(q["velocityx"] - ue).sum() * dx, ep=np.abs(q["pressure"] - pe).sum() * dx,
                xc=xc, xce=X0 + info["us"] * t)


def table(R, label):
    print(f"{label}")
    print(f"{'cells':>6} {'L1 density':>11} {'L1 velocity':>12} {'L1 pressure':>12} {'contact x':>10} {'exact':>8}")
    for r in R: print(f"{r['n']:6d} {r['er']:11.3e} {r['eu']:12.3e} {r['ep']:12.3e} {r['xc']:10.5f} {r['xce']:8.5f}")
    for a, b in zip(R[:-1], R[1:]):
        f = np.log(b["n"] / a["n"])
        print(f"  order {a['n']} -> {b['n']}: density {np.log(a['er'] / b['er']) / f:.2f}, velocity {np.log(a['eu'] / b['eu']) / f:.2f}, pressure {np.log(a['ep'] / b['ep']) / f:.2f}")


def tube(dirs, phases, outdir):
    R = sorted((tube_one(d) for d in dirs), key=lambda r: r["n"]); P = sorted((tube_one(d) for d in phases), key=lambda r: r["n"])
    i = R[0]["info"]
    print(f"exact: p* = {i['ps']:.6f}, u* = {i['us']:.6f}, shock speed {i['S']:.6f}, rho*_L {i['rsl']:.6f}, rho*_R {i['rsr']:.6f}")
    table(R, "two gases as SPECIES of one phase (species.eos = mixture)")
    if P: table(P, "two gases as the two PHASES")
    with open(os.path.join(outdir, "two_gamma_tube_summary.csv"), "w") as fo:
        fo.write("model,cells,L1_density,L1_velocity,L1_pressure,contact_x,contact_x_exact\n")
        for lab, S in (("species", R), ("phases", P)):
            for r in S: fo.write(f"{lab},{r['n']},{r['er']:.6g},{r['eu']:.6g},{r['ep']:.6g},{r['xc']:.6f},{r['xce']:.6f}\n")
    a = min(R, key=lambda r: abs(r["n"] - 400)); b = min(P, key=lambda r: abs(r["n"] - a["n"])) if P else None
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    xx = np.linspace(0, 1, 4000); re, ue, pe, _ = exact((xx - X0) / a["t"]); st = max(1, a["n"] // 100)
    for c, (ex, key, lab) in zip(ax[:3], ((re, "rho", "Density [kg/m$^3$]"), (ue, "u", "Velocity [m/s]"), (pe, "p", "Pressure [Pa]"))):
        c.axvspan(0, a["xce"], color="C0", alpha=0.06, lw=0); c.axvspan(a["xce"], 1, color="C1", alpha=0.06, lw=0)
        c.plot(xx, ex, "C3-", lw=1.5, label="Exact")
        c.plot(a["x"][::st], a[key][::st], "ko", ms=4, mfc="none", label="Numerical, Two Species")
        if b is not None: c.plot(b["x"][st // 2::st], b[key][st // 2::st], "s", color="0.45", ms=3.5, mfc="none", label="Numerical, Two Phases")
        c.set_xlabel("Position [m]"); c.set_ylabel(lab); c.set_title(lab.split(" [")[0]); c.set_xlim(0, 1); c.legend(fontsize=8)
    c = ax[3]; n = np.array([r["n"] for r in R], float)
    c.loglog(n, [100 * r["er"] for r in R], "ko-", mfc="none", label="Density, Two Species")
    c.loglog(n, [100 * r["ep"] for r in R], "ks--", mfc="none", label="Pressure, Two Species")
    if P:
        m = np.array([r["n"] for r in P], float)
        c.loglog(m, [100 * r["er"] for r in P], "o-", color="0.5", mfc="none", label="Density, Two Phases")
        c.loglog(m, [100 * r["ep"] for r in P], "s--", color="0.5", mfc="none", label="Pressure, Two Phases")
    c.loglog(n, 100 * R[0]["er"] * (n / n[0]) ** -1, "-", color="0.75", lw=1, label="First Order")
    c.set_xlabel("Cells"); c.set_ylabel("L$_1$ Error [%]"); c.set_title("Convergence"); c.legend(fontsize=7)
    c.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)); c.grid(which="minor", alpha=0.15)
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle(f"Shock Tube Between Two Gases ($\\gamma$ = 1.4 | 5/3), t = {a['t']:.2f} s, {a['n']} Cells"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "two_gamma_tube.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/two_gamma_tube.png")


def contact(dirs, outdir):
    R = []
    for d in dirs:
        pmax = umax = 0.0; hist = []
        for pf in files(d):
            t, x, q = load(pf, ["density", "velocityx", "pressure", "Y_A"])
            if t == 0.0: r0 = q["density"].copy(); m0 = q["density"].sum()
            hist.append((t, np.abs(q["pressure"] - 1).max(), np.abs(q["velocityx"] - 1).max()))
            pmax = max(pmax, hist[-1][1]); umax = max(umax, hist[-1][2])
        dx = x[1] - x[0]
        R.append(dict(d=d, n=len(x), t=t, x=x, p=q["pressure"], u=q["velocityx"], rho=q["density"], r0=r0, pmax=pmax, umax=umax,
                      er=np.abs(q["density"] - r0).sum() * dx, mass=q["density"].sum() / m0 - 1, hist=np.array(hist)))
    R.sort(key=lambda r: r["n"])
    print(f"{'cells':>6} {'max |p - 1|':>12} {'max |u - 1|':>12} {'L1 density (t = 1)':>19} {'mass':>10}")
    for r in R: print(f"{r['n']:6d} {100 * r['pmax']:10.3f} % {100 * r['umax']:10.3f} % {r['er']:19.3e} {r['mass']:+10.1e}")
    for a, b in zip(R[:-1], R[1:]):
        f = np.log(b["n"] / a["n"])
        print(f"  order {a['n']} -> {b['n']}: pressure error {np.log(a['pmax'] / b['pmax']) / f:.2f}, density {np.log(a['er'] / b['er']) / f:.2f}")
    with open(os.path.join(outdir, "isobaric_contact_summary.csv"), "w") as fo:
        fo.write("cells,max_abs_p_error_pct,max_abs_u_error_pct,L1_density,mass_drift\n")
        for r in R: fo.write(f"{r['n']},{100 * r['pmax']:.6g},{100 * r['umax']:.6g},{r['er']:.6g},{r['mass']:.3g}\n")
    a = R[len(R) // 2]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].plot(a["x"], a["r0"], "C3-", lw=1.5, label="Exact"); ax[0].plot(a["x"][::max(1, a["n"] // 100)], a["rho"][::max(1, a["n"] // 100)], "ko", ms=4, mfc="none", label="Numerical")
    ax[0].set_xlabel("Position [m]"); ax[0].set_ylabel("Density [kg/m$^3$]"); ax[0].set_title(f"Density After One Pass, {a['n']} Cells"); ax[0].legend()
    ax[1].axhline(1.0, color="C3", lw=1.5, label="Exact"); ax[1].plot(a["x"], a["p"], "k-", lw=0.9, label="Numerical, Pressure")
    ax[1].plot(a["x"], a["u"], "-", color="0.5", lw=0.9, label="Numerical, Velocity")
    ax[1].set_xlabel("Position [m]"); ax[1].set_ylabel("Pressure [Pa], Velocity [m/s]"); ax[1].set_title("Pressure and Velocity After One Pass"); ax[1].legend(fontsize=8)
    n = np.array([r["n"] for r in R], float); c = ax[2]
    c.loglog(n, [100 * r["pmax"] for r in R], "ko-", mfc="none", label="Pressure"); c.loglog(n, [100 * r["umax"] for r in R], "ks--", mfc="none", label="Velocity")
    c.loglog(n, 100 * R[0]["pmax"] * (n / n[0]) ** -1, "-", color="0.75", lw=1, label="First Order")
    c.set_xlabel("Cells"); c.set_ylabel("Maximum Error Over the Run [%]"); c.set_title("Pressure and Velocity Error at the Contact"); c.legend(fontsize=8)
    c.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)); c.grid(which="minor", alpha=0.15)
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle("Advected Isobaric Contact Between Two Gases ($\\gamma$ = 1.4 | 5/3) as Species of One Phase"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "isobaric_contact.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/isobaric_contact.png")


def main():
    args = sys.argv[1:]; mode = args[0]; rest = args[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = rest[rest.index("--outdir") + 1] if "--outdir" in rest else os.path.join(here, "Images", "TwoGamma")
    dirs, phases, tgt, skip = [], [], None, False
    for a in rest:
        if skip: skip = False; continue
        if a == "--outdir": skip = True; continue
        if a == "--phases": tgt = phases; continue
        (dirs if tgt is None else tgt).append(a)
    os.makedirs(outdir, exist_ok=True)
    tube(dirs, phases, outdir) if mode == "tube" else contact(dirs, outdir)


if __name__ == "__main__":
    main()
