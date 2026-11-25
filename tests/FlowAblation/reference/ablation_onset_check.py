#!/usr/bin/env python3
"""
Ablation-onset test (tests/FlowAblation/input_1D_AblationOnset): runs against the reference.

  python3 ablation_onset_check.py <run_dir> [<run_dir> ...] [--outdir DIR] [--ref-N NS]

Reference: reference/ablation_onset_reference.py (stage 1 closed form; stage 2 fine-grid front-fixing
solution, NS solid / 4 NS gas nodes, default NS = 1600).  Per run:
  onset time   -- the uncapped surface temperature the code computes at the face (flux continuity,
                  T_w* = (h_g T_g + h_s T_s)/(h_g + h_s), same distances and far-cell rule as Hydro2.cpp),
                  of the last three plot files before the surface leaves its face, extrapolated to T_abl;
  recession    -- s = x0 - (phi = 0.5 crossing), against the reference at every plot time after onset;
  temperatures -- gas T and solid ablation_Ts at the last plot time, error / (T_g - T_s0), away from the surface cell.
The first run's figure is written; with several runs, a convergence figure too.
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import yt
yt.funcs.mylog.setLevel(40)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_onset_reference as R

RHO, CP, TG = 1.0, 3.5, 1.0


def meta(d, key, default=None):
    val = default
    for ln in open(os.path.join(d, "output", "metadata")):
        m = re.match(r"\s*" + re.escape(key) + r"\s*=\s*(\S+)", ln)
        if m: val = float(m.group(1))
    if val is None: raise SystemExit(f"{key} not in metadata")
    return val


def params(d):
    kg = meta(d, "mu0") * CP / meta(d, "thermal.Pr")
    return dict(rho_g=RHO, cp_g=CP, k_g=kg, T_g=TG, rho_s=meta(d, "ablation.rho_s"), cp_s=meta(d, "ablation.cp_s"),
                k_s=meta(d, "ablation.k_s"), T_s0=meta(d, "ablation.T_solid0"), T_abl=meta(d, "ablation.T_wall"),
                Qstar=meta(d, "ablation.Qstar"), L=meta(d, "solid.phi.ic.expression.constant.xw"),
                X1=meta(d, "geometry.prob_hi"))


def load(d, par):
    out = []
    for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4])):
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; X1 = float(ds.domain_right_edge[0]); dx = X1 / n; x = (np.arange(n) + 0.5) * dx
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
        phi = q("phi"); i = np.where(phi >= 0.5)[0][0]
        xs = x[i - 1] + (x[i] - x[i - 1]) * (0.5 - phi[i - 1]) / (phi[i] - phi[i - 1]) if i > 0 else 0.0
        T, Ts = q("T"), q("ablation_Ts")
        # the surface temperature exactly as the gas conduction block computes it (Hydro2.cpp, abl_cond):
        # one-sided distances from phi, the far-cell rule on either side, flux continuity, then the cap
        phf, phs = phi[i], phi[i - 1]
        tt = min(max((phf - 0.5) / max(phf - phs, 1e-12), 0.0), 1.0)
        far = tt < 0.5 and phi[i + 1] >= 0.5
        Tg = T[i + 1] if far else T[i]; hg = par["k_g"] / ((1 + tt if far else max(tt, 0.25)) * dx)
        sfar = (1 - tt < 0.5) and i >= 2 and phi[i - 2] < 0.5
        Tsr = Ts[i - 2] if sfar else Ts[i - 1]; hs = par["k_s"] / ((2 - tt if sfar else max(1 - tt, 0.25)) * dx)
        Tw_star = (hg * Tg + hs * Tsr) / (hg + hs)
        sharp = phs < 1e-6 and phf > 1 - 1e-6           # surface still on its initial face (no recession yet)
        out.append(dict(t=float(ds.current_time), x=x, dx=dx, phi=phi, T=T, Ts=Ts, xs=xs, i=i, Tw=Tw_star, sharp=sharp,
                        qn=q("ablation_q").max()))
    return out


def analyse(d, ref_N):
    par = params(d); F = load(d, par)
    t = np.array([f["t"] for f in F]); s = par["L"] - np.array([f["xs"] for f in F]); Tw = np.array([f["Tw"] for f in F])
    sharp = np.array([f["sharp"] for f in F]); n = len(F[0]["x"]); dT = par["T_g"] - par["T_s0"]
    ta_cf = R.closed_form_onset(par)
    tref = t[t > 0.05]
    ref = R.solve(par, tref, Ns=ref_N, Ng=4 * ref_N)
    # numerical onset: T_w* of the last three plot files before the surface leaves its face, extrapolated
    # (quadratic in t) to T_abl.  The onset is ill-conditioned: dt_abl = dT_w / (dT_w/dt), and dT_w/dt is small there.
    k = np.where(sharp & (t > 0))[0]
    k = k[k < np.where(~sharp)[0][0]] if (~sharp).any() else k
    c = np.polyfit(t[k[-3:]], Tw[k[-3:]] - par["T_abl"], 2)
    rts = np.roots(c); rts = rts[np.isreal(rts)].real
    t_on = rts[np.argmin(np.abs(rts - t[k[-1]]))]
    t_first = t[np.where(~sharp)[0][0]] if (~sharp).any() else np.nan
    s_ref = np.interp(t, ref["t"], ref["s"], left=0.0)
    # first plot time at which the surface has left its initial face, and the largest recession before onset
    pre = t < ref["t_abl"]
    res = dict(t_first=t_first, d=d, n=n, dx=F[0]["dx"], par=par, F=F, t=t, s=s, Tw=Tw, sharp=sharp, ref=ref, ta_cf=ta_cf,
               t_on=t_on, s_ref=s_ref, s_pre=np.abs(s[pre]).max())
    # recession error: relative at the end, and the max over t >= t_abl + 1 relative to the local reference
    late = t >= ref["t_abl"] + 1.0
    res["s_err_end"] = s[-1] / s_ref[-1] - 1
    res["s_err_late"] = (s[late] - s_ref[late])[np.argmax(np.abs(s[late] - s_ref[late]))] / s_ref[-1]
    # temperature profile at the last plot time
    L = F[-1]; x = L["x"]; fl = L["phi"] >= 0.5; Tnum = np.where(fl, L["T"], L["Ts"])
    xr, Tr = ref["profile"](len(ref["t"]) - 1)
    Tex = np.interp(x, xr, Tr)
    ok = np.abs(x - L["xs"]) > 1.5 * L["dx"]
    err = np.abs(Tnum - Tex) / dT
    res.update(x=x, Tnum=Tnum, Tex=Tex, fl=fl, ok=ok, err=err, xr=xr, Tr=Tr, e_gas=err[ok & fl].max(), e_sol=err[ok & ~fl].max())
    # surface-temperature history before onset (sharp phi): against the closed form
    b = sharp & (t >= 0.5) & (t < ref["t_abl"])
    res["Tw_err"] = np.abs(Tw[b] - R.closed_form_Tw(par, t[b])).max() / dT
    print(f"{d}: {n} cells (dx = {L['dx']:.3e}), {len(F)} plot files to t = {t[-1]:.2f}")
    print(f"  onset: run {t_on:.5f}   reference {ref['t_abl']:.5f}   closed form {ta_cf:.5f}   ({100 * (t_on / ta_cf - 1):+.3f} %);"
          f"  first plot file with the surface off its face: t = {t_first:.3f}")
    print(f"  recession before onset (max) {res['s_pre']:.2e}  ({res['s_pre'] / L['dx']:.2e} cells)")
    print(f"  surface temperature before onset: max |T_w* - closed form| = {100 * res['Tw_err']:.3f} % of dT (0.5 <= t < t_abl)")
    print(f"  recession at t = {t[-1]:.2f}: {s[-1]:.6f}  reference {s_ref[-1]:.6f}  ({100 * res['s_err_end']:+.3f} %);"
          f"  largest deviation for t >= t_abl + 1: {100 * res['s_err_late']:+.3f} % of s(t_end)")
    print(f"  temperature profile at t = {t[-1]:.2f}: max error gas {100 * res['e_gas']:.3f} %, solid {100 * res['e_sol']:.3f} % of dT")
    return res


STYLE = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14, "legend.fontsize": 13,
         "xtick.labelsize": 13, "ytick.labelsize": 13}
NUM = dict(color="C3", lw=0, marker="o", mfc="none", ms=6, label="Numerical")


def figure(r, outdir):
    plt.rcParams.update(STYLE)
    par, ref, t = r["par"], r["ref"], r["t"]; ta = ref["t_abl"]
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.6))
    # (a) surface temperature history
    a = ax[0, 0]; tt = np.linspace(1e-3, t[-1], 600)
    Twr = np.where(tt < ta, R.closed_form_Tw(par, tt), par["T_abl"])
    a.plot(tt, Twr, "k-", lw=1.8, label="Reference")
    m = t > 0
    a.plot(t[m], np.minimum(r["Tw"][m], par["T_abl"]), markevery=max(1, m.sum() // 24), **NUM)
    a.axhline(par["T_abl"], color="0.4", ls="--", lw=1.0); a.text(0.03 * t[-1], par["T_abl"] + 0.0001, r"$T_{abl}$", color="0.3"); a.text(ta, par["T_s0"] + 0.47 * (par["T_g"] - par["T_s0"]), r" $t_{abl}$", color="0.3")
    a.axvline(ta, color="0.4", ls=":", lw=1.2)
    a.set_xlabel("Time, $t$"); a.set_ylabel("Surface Temperature, $T_w$"); a.set_title("Surface Temperature")
    a.ticklabel_format(axis="y", useOffset=False); a.set_ylim(par["T_s0"] + 0.45 * (par["T_g"] - par["T_s0"]), par["T_abl"] + 0.0006)
    a.legend(frameon=False, loc="lower right")
    # (b) recession
    a = ax[0, 1]
    a.plot(ref["t"], ref["s"], "k-", lw=1.8, label="Reference")
    a.plot(t, r["s"], markevery=max(1, len(t) // 20), **NUM)
    a.axvline(ta, color="0.4", ls=":", lw=1.2); a.text(ta, 0.55 * ref["s"].max(), r" $t_{abl}$", color="0.3")
    a.set_xlabel("Time, $t$"); a.set_ylabel("Surface Recession, $s$"); a.set_title("Surface Recession"); a.legend(frameon=False, loc="upper left")
    # (c) temperature profiles at the last time
    a = ax[1, 0]; x = r["x"]; X0 = par["L"]
    win = x <= X0 + 0.25
    a.axvspan(0, r["F"][-1]["xs"], color="#E0E0E0", lw=0, zorder=0)
    wr = r["xr"] <= X0 + 0.25
    a.plot(r["xr"][wr], r["Tr"][wr], "k-", lw=1.8, label="Reference")
    a.plot(x[win], r["Tnum"][win], markevery=max(1, win.sum() // 22), **NUM)
    a.axhline(par["T_abl"], color="0.4", ls="--", lw=1.0)
    a.set_xlim(0, X0 + 0.25); a.set_xlabel("Position, $x$"); a.set_ylabel("Temperature, $T$")
    a.ticklabel_format(axis="y", useOffset=False); a.legend(frameon=False, loc="lower right")
    a.set_title(f"Temperature Profile, $t$ = {t[-1]:.1f}")
    # (d) profile error
    a = ax[1, 1]; ok = r["ok"] & win
    st = max(1, ok.sum() // 40)
    a.semilogy(x[ok][::st], 100 * np.maximum(r["err"][ok][::st], 1e-12), "C3o", mfc="none", ms=5)
    a.axvspan(0, r["F"][-1]["xs"], color="#E0E0E0", lw=0, zorder=0)
    a.set_xlim(0, X0 + 0.25); a.set_xlabel("Position, $x$"); a.set_ylabel(r"Error, $|T - T_{ref}|/\Delta T$ [%]")
    a.set_title("Temperature Error"); a.set_ylim(bottom=1e-4)
    a.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)); a.grid(which="minor", alpha=0.15)
    for a in ax.ravel(): a.grid(alpha=0.3)
    fig.suptitle(f"Ablation Onset: Surface Heated to $T_{{abl}}$ ({r['n']} Cells)", fontsize=16); fig.tight_layout()
    for ext in ("png", "eps"): fig.savefig(os.path.join(outdir, f"ablation_onset_N{r['n']}." + ext), dpi=200)
    plt.close(fig)
    print(f"  wrote {outdir}/ablation_onset_N{r['n']}.png")


def convergence(rs, outdir):
    plt.rcParams.update(STYLE)
    rs = sorted(rs, key=lambda r: r["n"])
    n = np.array([r["n"] for r in rs]); dx = np.array([r["dx"] for r in rs])
    e_on = np.array([abs(r["t_on"] / r["ta_cf"] - 1) for r in rs]); e_s = np.array([abs(r["s_err_end"]) for r in rs])
    e_T = np.array([max(r["e_gas"], r["e_sol"]) for r in rs])
    print("\n  cells   dx         onset err %   recession err % (t_end)   profile err % dT")
    for k in range(len(rs)):
        print(f"  {n[k]:5d}  {dx[k]:.3e}   {100 * e_on[k]:9.4f}     {100 * e_s[k]:9.4f}                 {100 * e_T[k]:8.4f}")
        if k: print(f"         observed order: onset {np.log(e_on[k - 1] / e_on[k]) / np.log(dx[k - 1] / dx[k]):.2f}, "
                    f"recession {np.log(e_s[k - 1] / e_s[k]) / np.log(dx[k - 1] / dx[k]):.2f}, profile {np.log(e_T[k - 1] / e_T[k]) / np.log(dx[k - 1] / dx[k]):.2f}")
    fig, a = plt.subplots(figsize=(9.0, 5.0))
    a.loglog(dx, 100 * e_on, "C3o-", mfc="none", ms=7, label="Onset Time, $t_{abl}$")
    a.loglog(dx, 100 * e_s, "C0s-", mfc="none", ms=7, label="Recession, $s(t_{end})$")
    a.loglog(dx, 100 * e_T, "C2^-", mfc="none", ms=7, label="Temperature Profile")
    for p, ls in ((1, "--"), (2, ":")):
        a.loglog(dx, 100 * e_s[0] * (dx / dx[0]) ** p, "k" + ls, lw=1.0, label=f"Order {p}")
    a.set_xlabel(r"Cell Size, $\Delta x$"); a.set_ylabel("Error [%]"); a.set_title("Grid Convergence, Ablation Onset")
    a.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(1.02, 1.0)); a.grid(which="both", alpha=0.3); fig.tight_layout()
    for ext in ("png", "eps"): fig.savefig(os.path.join(outdir, "ablation_onset_convergence." + ext), dpi=200)
    plt.close(fig)
    print(f"  wrote {outdir}/ablation_onset_convergence.png")


def main():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "Images", "R32_Onset"); ref_N = 1600
    if "--outdir" in args: k = args.index("--outdir"); outdir = args[k + 1]; del args[k:k + 2]
    if "--ref-N" in args: k = args.index("--ref-N"); ref_N = int(args[k + 1]); del args[k:k + 2]
    os.makedirs(outdir, exist_ok=True)
    rs = [analyse(d, ref_N) for d in args]
    for r in rs: figure(r, outdir)
    if len(rs) > 1: convergence(rs, outdir)


if __name__ == "__main__":
    main()
