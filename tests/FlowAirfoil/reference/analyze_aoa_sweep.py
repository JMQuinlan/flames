#!/usr/bin/env python3
"""
NACA 4-digit angle-of-attack sweep analysis (runs made by TEMPLATES/sweep_aoa.bash).

  python3 analyze_aoa_sweep.py <sweep_root> [--outdir DIR] [--avg-from T] [--ref CSV] [--ref-label TEXT]

--ref CSV: published data to overlay (columns Re,alpha,Cl,Cd[,kind]; '#' comments), rows with
the sweep's Re are used, e.g. data/naca0008_lowRe_reference.csv.  The airfoil (NACA code) is
read from the case inputs; file names use it (naca<code>_*.png).  The high-Re NACA 0012
experimental lines are drawn only for the 0012 without --ref.

<sweep_root> holds aoa_<AoA>/ case folders (input, out/ plotfiles, out_forces.dat).
Cases are auto-detected; cases still running are analysed with what exists and
flagged in the CSV.  Mach / Re / max_level are read from each case's input.

Figures (same style as the dissertation Chapter 4 figures) -> <outdir>
(default Images/<sweep_root name>/ next to this script):
  naca0012_polars_sweep.png     Cl, Cd (mean +- std over the averaging window) and
                                Cl/Cd vs AoA; thin-airfoil 2 pi alpha / beta
                                (Prandtl-Glauert, beta = sqrt(1 - M^2)), the
                                high-Re experimental slope 0.11/deg and Cd0 ~ 0.0065,
                                and the pressure-only force (grey)
  naca0012_Cl_sweep.png / naca0012_Cd_sweep.png          single panels
  naca0012_FirstCl_sweep.png / naca0012_FirstCd_sweep.png  forces at t = T_FIRST
                                (impulsive start, before the wake develops)
  naca0012_history_sweep.png    Cl(t), Cd(t) for every AoA (grey scale + line styles)
  naca0012_fields_sweep.png     Cp + streamlines, rows = FIELD_AOAS, columns = FIELD_TIMES
  aoa_sweep_summary.csv         per-AoA metrics (+ Mach, Re, level, commits)
Force coefficients: C = F / (0.5 rho U^2 c), rho = 1, c = 1, freestream along +x,
so Cd = Fx / q and Cl = Fy / q (the airfoil, not the flow, is rotated).
F_total = momentum pushed into the (inert, mirrored-wall) solid = pressure + viscous;
F_pressure = sum of -p n dA over the wall faces.
"""
import glob, os, re, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== CONFIG ======================================
RHO, CHORD = 1.0, 1.0
AVG_FRACTION   = 1.0 / 3.0        # average over the last third of the run (unless --avg-from)
T_FIRST        = 0.5              # "first" forces (impulsive start)
EXP_SLOPE      = 0.11             # high-Re NACA 0012 lift slope [1/deg] (Abbott & von Doenhoff)
EXP_CD0        = 0.0065           # high-Re NACA 0012 Cd0
FIELD_AOAS     = None            # None = up to 6 evenly spaced AoA from the cases present; or a list
FIELD_TIMES    = [0.5, "mid", "end"]
FIELD_WINDOW   = [-0.8, 1.4, -0.6, 0.6]    # xlo, xhi, ylo, yhi
FIELD_RES      = 500                       # pixels in x
CP_LIM         = 2.0
DPI            = 150
# ============================================================================


def case_info(d):
    """AoA, Mach, Re, max_level, stop_time from the case input."""
    txt = open(os.path.join(d, "input"), errors="ignore").read()
    m = re.search(r"M = ([0-9.eE+-]+), Re = ([0-9.eE+-]+), AoA = ([0-9.eE+-]+) deg", txt)
    af = re.search(r"Laminar NACA (\d{4})", txt)
    lvl = int(re.search(r"^amr\.max_level\s*=\s*(\d+)", txt, re.M).group(1))
    stop = float(re.search(r"^stop_time\s*=\s*([0-9.eE+-]+)", txt, re.M).group(1))
    return dict(aoa=float(m.group(3)), mach=float(m.group(1)), re=float(m.group(2)), lvl=lvl, stop=stop,
                airfoil=af.group(1) if af else "0012")


def load_forces(d, mach):
    f = os.path.join(d, "out_forces.dat")
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        return None
    F = np.loadtxt(f, comments="#", ndmin=2)
    F = F[F[:, 1] == F[:, 1].max()]
    q = 0.5 * RHO * mach**2 * CHORD
    return dict(t=F[:, 2], Cd=F[:, 3] / q, Cl=F[:, 4] / q, Cdp=F[:, 5] / q, Clp=F[:, 6] / q)


def strouhal(t, y, U):
    if len(t) < 64 or np.std(y) < 1e-3:
        return None
    tu = np.linspace(t[0], t[-1], 4096); yu = np.interp(tu, t, y) - np.mean(y)
    F = np.abs(np.fft.rfft(yu * np.hanning(len(yu)))); f = np.fft.rfftfreq(len(yu), tu[1] - tu[0])
    k = 1 + np.argmax(F[1:])
    return f[k] * CHORD / U


def git_head():
    try:
        return subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "describe", "--always", "--dirty"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def run_commit(d):
    meta = os.path.join(d, "out", "metadata")
    if os.path.exists(meta):
        for line in open(meta, errors="ignore"):
            if line.strip().startswith("Git_commit_hash"):
                return line.split("=", 1)[1].strip()
    return ""


def plotfiles(d):
    pfs = [p for p in glob.glob(os.path.join(d, "out", "*cell")) if ".old." not in p]
    return sorted(pfs, key=lambda p: int(re.search(r"(\d+)cell$", p).group(1)))


def pf_time(p):
    h = open(os.path.join(p, "Header")).read().split("\n"); nf = int(h[1]); return float(h[3 + nf])


def field_panel(ax, pf, mach, extent):
    """Cp + streamlines on `ax` from plotfile `pf` (finest level, interpolated)."""
    import yt
    from scipy.interpolate import RegularGridInterpolator
    yt.set_log_level(40)
    ds = yt.load(pf); L = ds.index.max_level
    xlo, xhi, ylo, yhi = extent
    dxf = (ds.domain_width / (ds.domain_dimensions * ds.refine_by**L)).to_value()
    le = ds.domain_left_edge.to_value()
    i0 = int(np.floor((xlo - le[0]) / dxf[0])) - 1; i1 = int(np.ceil((xhi - le[0]) / dxf[0])) + 1
    j0 = int(np.floor((ylo - le[1]) / dxf[1])) - 1; j1 = int(np.ceil((yhi - le[1]) / dxf[1])) + 1
    left = [le[0] + i0 * dxf[0], le[1] + j0 * dxf[1], le[2]]
    try:
        cg = ds.smoothed_covering_grid(level=L, left_edge=left, dims=[i1 - i0, j1 - j0, 1])
        get = lambda f: np.asarray(cg[("boxlib", f)])[:, :, 0]
        fields = {f: get(f) for f in ("pressure", "velocityx", "velocityy", "phi")}
    except RuntimeError:
        ds.force_periodicity()
        cg = ds.smoothed_covering_grid(level=L, left_edge=left, dims=[i1 - i0, j1 - j0, 1])
        fields = {f: np.asarray(cg[("boxlib", f)])[:, :, 0] for f in ("pressure", "velocityx", "velocityy", "phi")}
    xc = left[0] + (np.arange(i1 - i0) + 0.5) * dxf[0]; yc = left[1] + (np.arange(j1 - j0) + 0.5) * dxf[1]
    nx = FIELD_RES; ny = int(nx * (yhi - ylo) / (xhi - xlo))
    x = np.linspace(xlo, xhi, nx); y = np.linspace(ylo, yhi, ny)
    X, Y = np.meshgrid(x, y)
    interp = lambda a: RegularGridInterpolator((xc, yc), a)(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    p, u, v, phi = (interp(fields[f]) for f in ("pressure", "velocityx", "velocityy", "phi"))
    Pinf = 1.0 / 1.4
    cp = (p - Pinf) / (0.5 * RHO * mach**2)
    solid = phi < 0.5
    cpm = np.ma.masked_where(solid, cp)
    im = ax.imshow(cpm, origin="lower", extent=extent, cmap="coolwarm", vmin=-CP_LIM, vmax=CP_LIM, aspect="equal")
    um, vm = np.ma.masked_where(solid, u), np.ma.masked_where(solid, v)
    ax.streamplot(x, y, um, vm, density=1.1, color="0.15", linewidth=0.6, arrowsize=0.6)
    ax.contourf(X, Y, phi, levels=[-1, 0.5], colors="k")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    return im, float(ds.current_time)


def main():
    args = sys.argv[1:]
    root = os.path.abspath(args[0])
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (args[args.index("--outdir") + 1] if "--outdir" in args
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(root))))
    avg_from = float(args[args.index("--avg-from") + 1]) if "--avg-from" in args else None
    ref_path = args[args.index("--ref") + 1] if "--ref" in args else None
    ref_label = args[args.index("--ref-label") + 1] if "--ref-label" in args else None
    os.makedirs(outdir, exist_ok=True)

    cases = []
    for d in sorted(glob.glob(os.path.join(root, "aoa_*"))):
        if not os.path.exists(os.path.join(d, "input")):
            continue
        ci = case_info(d); fo = load_forces(d, ci["mach"])
        if fo is None:
            continue
        ci.update(dir=d, forces=fo); cases.append(ci)
    cases.sort(key=lambda c: c["aoa"])
    if not cases:
        print("no cases with a force history under", root); return
    M, Re, lvl, AF = cases[0]["mach"], cases[0]["re"], cases[0]["lvl"], cases[0]["airfoil"]
    ref = None
    if ref_path:
        import csv
        R = [r for r in csv.DictReader(l for l in open(ref_path) if l.strip() and not l.startswith("#"))
             if abs(float(r["Re"]) - Re) < 0.5]
        ref = dict(alpha=np.array([float(r["alpha"]) for r in R]), Cl=np.array([float(r["Cl"]) for r in R]),
                   Cd=np.array([float(r["Cd"]) for r in R]))
        ref_label = ref_label or f"reference (NACA {AF}, Re {Re:g})"
    show_exp = (AF == "0012") and ref is None
    beta = np.sqrt(1 - M**2)
    print(f"{root}: {len(cases)} cases  M = {M}  Re = {Re:g}  max_level = {lvl}")

    rows = []
    for c in cases:
        fo = c["forces"]; t = fo["t"]; stop = c["stop"]
        t0 = avg_from if avg_from is not None else stop * (1 - AVG_FRACTION)
        m = t >= t0
        done = t[-1] >= stop - 1e-6
        if m.sum() < 10:                     # still before the window: use the last 10% of what exists
            m = t >= t[-1] - 0.1 * max(t[-1], 1e-9)
        k1 = int(np.argmin(np.abs(t - T_FIRST)))
        r = dict(aoa=c["aoa"], t_end=float(t[-1]), stop=stop, complete=done, avg_from=float(t[m][0]),
                 Cl_mean=float(fo["Cl"][m].mean()), Cl_std=float(fo["Cl"][m].std()),
                 Cd_mean=float(fo["Cd"][m].mean()), Cd_std=float(fo["Cd"][m].std()),
                 Clp_mean=float(fo["Clp"][m].mean()), Cdp_mean=float(fo["Cdp"][m].mean()),
                 Cl_first=float(fo["Cl"][k1]), Cd_first=float(fo["Cd"][k1]),
                 Cl_thin=float(2 * np.pi / beta * np.radians(c["aoa"])),
                 St=(strouhal(t[m], fo["Cl"][m], M) if done else None),   # not from a transient
                 run_commit=run_commit(c["dir"]))
        r["L_over_D"] = r["Cl_mean"] / r["Cd_mean"] if r["Cd_mean"] else None
        r["Cl_ref"] = r["Cd_ref"] = r["Cl_err_pct"] = r["Cd_err_pct"] = None
        if ref is not None and np.any(np.abs(ref["alpha"] - c["aoa"]) < 1e-6):
            k = int(np.argmin(np.abs(ref["alpha"] - c["aoa"])))
            r["Cl_ref"], r["Cd_ref"] = float(ref["Cl"][k]), float(ref["Cd"][k])
            if abs(r["Cl_ref"]) > 1e-9: r["Cl_err_pct"] = 100 * (r["Cl_mean"] - r["Cl_ref"]) / r["Cl_ref"]
            r["Cd_err_pct"] = 100 * (r["Cd_mean"] - r["Cd_ref"]) / r["Cd_ref"]
        rows.append(r)
        print(f"  AoA {c['aoa']:4.1f}: t {t[-1]:6.2f}/{stop:g}{'' if done else ' (running)'}  "
              f"Cl {r['Cl_mean']:+.4f} +- {r['Cl_std']:.4f}  Cd {r['Cd_mean']:.4f} +- {r['Cd_std']:.4f}  "
              f"thin {r['Cl_thin']:.4f}" + (f"  ref Cl {r['Cl_ref']:.4f} Cd {r['Cd_ref']:.4f}" if r["Cl_ref"] is not None else "")
              + (f"  St {r['St']:.3f}" if r["St"] else ""))

    A = np.array([r["aoa"] for r in rows])
    CL = np.array([r["Cl_mean"] for r in rows]); CLs = np.array([r["Cl_std"] for r in rows])
    CD = np.array([r["Cd_mean"] for r in rows]); CDs = np.array([r["Cd_std"] for r in rows])
    CLP = np.array([r["Clp_mean"] for r in rows]); CDP = np.array([r["Cdp_mean"] for r in rows])
    aa = np.linspace(0, max(A.max(), 1), 200)
    thin = 2 * np.pi / beta * np.radians(aa)
    nrun = sum(not r["complete"] for r in rows)
    title = (rf"NACA {AF},  $M_\infty = {M:.2f}$,  $Re = {Re:g}$   ({len(rows)} cases"
             + (f", {nrun} still running" if nrun else "") + ")")
    wlab = rf"$t \geq {rows[0]['avg_from']:.0f}$"

    def cl_axes(ax, legend=True):
        ax.plot(A, CLP, "s--", color="0.45", ms=5, label="pressure only")
        ax.plot(aa, thin, "r-", lw=1.2, label=r"thin airfoil $2\pi\alpha/\beta$")
        if show_exp:
            ax.plot(aa, EXP_SLOPE * aa, "k:", lw=1.2, label=rf"experiment ${EXP_SLOPE}/^\circ$ (high Re)")
        if ref is not None:
            ax.plot(ref["alpha"], ref["Cl"], "D", mfc="white", mec="k", ms=6, label=ref_label)
        ax.errorbar(A, CL, yerr=CLs, fmt="o-", color="k", ms=5, capsize=3, label=rf"present ($\bar C_l \pm \sigma$, {wlab})")
        ax.set_xlabel(r"$\alpha$ [$^\circ$]"); ax.set_ylabel(r"$\bar C_l$"); ax.grid(alpha=0.3)
        ax.set_xticks(np.arange(0, A.max() + 1, 2))
        if legend: ax.legend(fontsize=9, frameon=False)

    def cd_axes(ax, legend=True):
        ax.plot(A, CDP, "s--", color="0.45", ms=5, label="pressure only")
        if show_exp:
            ax.axhline(EXP_CD0, color="k", ls=":", lw=1.2, label=rf"experiment $C_{{d0}} \approx {EXP_CD0}$ (high Re)")
        if ref is not None:
            ax.plot(ref["alpha"], ref["Cd"], "D", mfc="white", mec="k", ms=6, label=ref_label)
        ax.errorbar(A, CD, yerr=CDs, fmt="o-", color="k", ms=5, capsize=3, label="present")
        ax.set_xlabel(r"$\alpha$ [$^\circ$]"); ax.set_ylabel(r"$\bar C_d$"); ax.grid(alpha=0.3)
        ax.set_xticks(np.arange(0, A.max() + 1, 2))
        if legend: ax.legend(fontsize=9, frameon=False)

    # ---- polars (3 panels) ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    cl_axes(ax[0]); cd_axes(ax[1])
    ax[2].plot(A, CL / CD, "o-", color="k", ms=5)
    ax[2].set_xlabel(r"$\alpha$ [$^\circ$]"); ax[2].set_ylabel(r"$\bar C_l / \bar C_d$"); ax[2].grid(alpha=0.3)
    ax[2].set_xticks(np.arange(0, A.max() + 1, 2))
    fig.suptitle(title); fig.tight_layout(); fig.savefig(os.path.join(outdir, f"naca{AF}_polars_sweep.png"), dpi=DPI); plt.close(fig)
    # ---- single panels ----
    for name, fn in (("Cl", cl_axes), ("Cd", cd_axes)):
        fig, ax1 = plt.subplots(figsize=(5.6, 5.6)); fn(ax1); ax1.set_title(title.split("   (")[0], fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"naca{AF}_{name}_sweep.png"), dpi=DPI); plt.close(fig)
    # ---- first (impulsive-start) forces ----
    for name, key, ylab in (("Cl", "Cl_first", rf"$C_l$ at $t = {T_FIRST}$"), ("Cd", "Cd_first", rf"$C_d$ at $t = {T_FIRST}$")):
        fig, ax1 = plt.subplots(figsize=(5.6, 5.6))
        ax1.plot(A, [r[key] for r in rows], "o", color="k", ms=6, label=rf"present ($t = {T_FIRST}$)")
        if name == "Cl":
            ax1.plot(aa, thin, "r-", lw=1.2, label=r"thin airfoil $2\pi\alpha/\beta$")
            if show_exp:
                ax1.plot(aa, EXP_SLOPE * aa, "k:", lw=1.2, label=rf"experiment ${EXP_SLOPE}/^\circ$ (high Re)")
        ax1.set_xlabel(r"$\alpha$ [$^\circ$]"); ax1.set_ylabel(ylab); ax1.grid(alpha=0.3)
        ax1.set_xticks(np.arange(0, A.max() + 1, 2)); ax1.legend(fontsize=9, frameon=False)
        ax1.set_title(title.split("   (")[0], fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"naca{AF}_First{name}_sweep.png"), dpi=DPI); plt.close(fig)
    # ---- force histories ----
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True)
    for i, c in enumerate(cases):
        g = 0.75 * (1 - i / max(len(cases) - 1, 1))
        fo = c["forces"]
        for k, key in enumerate(("Cl", "Cd")):
            ax[k].plot(fo["t"], fo[key], color=str(g), ls=styles[i % len(styles)], lw=1.1,
                       label=rf"$\alpha = {c['aoa']:g}^\circ$")
    for k, lab in enumerate((r"$C_l$", r"$C_d$")):
        ax[k].axvline(rows[0]["avg_from"], color="k", ls=":", lw=0.8)
        ax[k].text(rows[0]["avg_from"], 1.0, " averaging window", transform=ax[k].get_xaxis_transform(),
                   fontsize=8, va="bottom", color="0.3")
        ax[k].set_ylabel(lab); ax[k].grid(alpha=0.3)
        ax[k].legend(ncol=3, fontsize=8, frameon=False, loc="upper right")
    ax[1].set_xlabel(r"$t$")
    fig.suptitle(rf"NACA {AF} force histories ($M_\infty = {M:.2f}$, $Re = {Re:g}$)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"naca{AF}_history_sweep.png"), dpi=DPI); plt.close(fig)
    # ---- fields: Cp + streamlines ----
    avail = [c for c in cases if plotfiles(c["dir"])]
    if FIELD_AOAS is None:
        idx = sorted(set(np.linspace(0, len(avail) - 1, min(6, len(avail))).round().astype(int))) if avail else []
        field_cases = [avail[i] for i in idx]
    else:
        field_cases = [c for c in avail if int(round(c["aoa"])) in FIELD_AOAS]
    if field_cases:
        stop = max(c["stop"] for c in field_cases)
        times = [stop / 2 if ft == "mid" else stop if ft == "end" else ft for ft in FIELD_TIMES]
        nr, nc = len(field_cases), len(times)
        fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc + 1.2, 2.1 * nr + 0.6), sharex=True, sharey=True, squeeze=False)
        im = None
        for i, c in enumerate(field_cases):
            pfs = plotfiles(c["dir"]); tt = np.array([pf_time(p) for p in pfs])
            for j, tw in enumerate(times):
                a = axs[i, j]
                k = int(np.argmin(np.abs(tt - tw)))
                if abs(tt[k] - tw) > 0.26:              # not reached yet
                    a.text(0.5, 0.5, "not yet run", ha="center", va="center", transform=a.transAxes, color="0.4")
                else:
                    im, tact = field_panel(a, pfs[k], c["mach"], FIELD_WINDOW)
                if i == 0: a.set_title(rf"$t = {tw:g}$")
                if j == 0: a.set_ylabel(rf"$\alpha = {c['aoa']:g}^\circ$" + "\n$y$")
                if i == nr - 1: a.set_xlabel("$x$")
        if im is not None:
            fig.subplots_adjust(right=0.88)
            cax = fig.add_axes([0.9, 0.2, 0.025, 0.6])
            cb = fig.colorbar(im, cax=cax, extend="both"); cb.set_label(r"$C_p$")
        fig.savefig(os.path.join(outdir, f"naca{AF}_fields_sweep.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)
    # ---- CSV ----
    cols = ["aoa", "complete", "t_end", "stop", "avg_from", "Cl_mean", "Cl_std", "Cd_mean", "Cd_std",
            "Clp_mean", "Cdp_mean", "L_over_D", "Cl_thin", "Cl_ref", "Cd_ref", "Cl_err_pct", "Cd_err_pct",
            "Cl_first", "Cd_first", "St", "run_commit"]
    with open(os.path.join(outdir, "aoa_sweep_summary.csv"), "w") as f:
        f.write(f"# NACA {AF} AoA sweep  Mach={M}  Re={Re:g}  max_level={lvl}  beta={beta:.4f}  "
                f"analysis_commit={git_head()}  sweep_root={root}\n")
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
    print(f"  wrote figures + aoa_sweep_summary.csv to {outdir}")


if __name__ == "__main__":
    main()
