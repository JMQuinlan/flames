#!/usr/bin/env python3
"""
Finite-strain uniaxial shock in a solid (tests/FlowElasticSolid/input_1D_Piston).

  python3 piston_check.py <run_dir> [U] [--outdir DIR]
  python3 piston_check.py <run_dir_U0.5> <run_dir_U1.0> <run_dir_U2.0> [--outdir DIR] [--no-gif]
      several speeds on one figure (U read from "_U<value>" in the directory names): piston_profiles.png,
      piston_hugoniot.png and Contours/piston.gif (needs frequent plot files: amr.plot_dt=0.002) in Images/Piston_combined/

Exact Rankine-Hugoniot state behind the shock for the model's equation of state
(Favrie, Gavrilyuk & Saurel, JCP 228, 2009, eqs. 9-12), uniaxial compression
a = rho1/rho0, Gt = diag(a^(4/3), a^(-2/3), a^(-2/3)):
    S11(a) = -mu a [ a^(8/3) - a^(4/3) - (J2 - J1)/3 ],  J1 = tr Gt, J2 = tr Gt^2
    e_e(a) = mu/(4 rho0) (J2 - 2 J1 + 3),   e_h = (p + gamma p_inf)/((gamma - 1) rho)
    s = U/(a - 1),  p1 - S11 - p0 = rho0 a U^2/(a - 1),
    e1 - e0 = (1/2)(p1 - S11 + p0)(1 - 1/a)/rho0
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)

RHO0, GAM, PINF, P0, MU = 1.0, 3.4, 10.0, 0.1, 12.0


def dev(a):
    g = np.array([a**(4 / 3), a**(-2 / 3), a**(-2 / 3)])
    J1, J2 = g.sum(), (g**2).sum()
    S = -MU * a * (g**2 - g - (J2 - J1) / 3)
    return S, MU / (4 * RHO0) * (J2 - 2 * J1 + 3)


def hugoniot(U):
    eh = lambda p, rho: (p + GAM * PINF) / ((GAM - 1) * rho)
    def f(a):
        S, ee = dev(a)
        p1 = P0 + S[0] + RHO0 * a * U * U / (a - 1)
        return eh(p1, RHO0 * a) + ee - eh(P0, RHO0) - 0.5 * (p1 - S[0] + P0) * (1 - 1 / a) / RHO0
    lo, hi = 1 + 1e-9, 3.0
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (flo > 0): lo = mid
        else: hi = mid
    a = 0.5 * (lo + hi); S, ee = dev(a)
    return dict(a=a, s=U / (a - 1), p=P0 + S[0] + RHO0 * a * U * U / (a - 1), S11=S[0], S22=S[1], W=RHO0 * a * ee)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    d = args[0]; U = float(args[1]) if len(args) > 1 and "--outdir" not in sys.argv[2:3] else 1.0
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    R = hugoniot(U)
    pf = sorted(glob.glob(os.path.join(d, "output", "*cell")))[-1]
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; x = -1 + (np.arange(n) + 0.5) * 2 / n; t = float(ds.current_time)
    g = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
    u, p, rho, Sxx, Syy, W = g("velocityx"), g("pressure"), g("density"), g("elastic_Sxx"), g("elastic_Syy"), g("elastic_W")
    i = np.where((rho[n // 2:] - RHO0) > 0.5 * (R["a"] - 1) * RHO0)[0][-1] + n // 2          # right shock: half-height
    xs = x[i] + (x[i + 1] - x[i]) * (rho[i] - RHO0 * (1 + 0.5 * (R["a"] - 1))) / (rho[i] - rho[i + 1])
    m = (x > 0.25 * xs) & (x < 0.75 * xs)          # shocked plateau, away from the wall-heating spike at x = 0
    rows = [("compression rho1/rho0", rho[m].mean() / RHO0, R["a"]), ("shock speed", xs / t, R["s"]),
            ("pressure p1", p[m].mean(), R["p"]), ("S_11", Sxx[m].mean(), R["S11"]), ("S_22", Syy[m].mean(), R["S22"]),
            ("normal stress -sigma_11", (p - Sxx)[m].mean(), R["p"] - R["S11"]), ("elastic energy W", W[m].mean(), R["W"])]
    print(f"{d}: U = {U:g}, t = {t:.4f};  max |u| behind the shock {np.abs(u[m]).max():.2e}")
    with open(os.path.join(outdir, "piston_summary.csv"), "w") as fo:
        fo.write("quantity,Hydro2,exact,error_pct\n")
        for k, a, b in rows:
            print(f"  {k:26s} {a:11.5f}   exact {b:11.5f}   ({100 * (a / b - 1):+.2f} %)")
            fo.write(f"{k},{a:.7g},{b:.7g},{100 * (a / b - 1):.3f}\n")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for a_, q, lab, ref in ((ax[0, 0], rho, r"$\rho$", R["a"] * RHO0), (ax[0, 1], p, r"$p$", R["p"]),
                            (ax[1, 0], Sxx, r"$S_{11}$", R["S11"]), (ax[1, 1], u, r"$u$", 0.0)):
        a_.plot(x, q, "k-", lw=1); a_.axhline(ref, color="C3", ls="--", lw=0.9, label="Rankine-Hugoniot")
        for sgn in (-1, 1): a_.axvline(sgn * R["s"] * t, color="0.6", ls=":")
        a_.set_ylabel(lab); a_.grid(alpha=0.3); a_.legend(fontsize=8)
    for a_ in ax[1]: a_.set_xlabel("x")
    fig.suptitle(f"Uniaxial shock in a solid, U = {U:g}: compression {100 * (R['a'] - 1):.1f} %, t = {t:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "piston.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/piston.png and piston_summary.csv")


def load(pf):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; x = -1 + (np.arange(n) + 0.5) * 2 / n
    g = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
    return float(ds.current_time), x, dict(rho=g("density"), p=g("pressure"), S11=g("elastic_Sxx"), u=g("velocityx"))


def combined(dirs, outdir, make_gif=True):
    """several piston speeds on one figure (U from "_U<value>" in each directory name):
      piston_profiles.png   rho, p, S_11, u at the final time, one colour per U, exact post-shock states dashed
      piston_hugoniot.png   shock speed, compression and normal stress against U: exact curves, numerical points
      Contours/piston.gif   schlieren strip (|d rho/dx|) of each run above the density profiles, in time"""
    import re
    runs = []
    for d in dirs:
        U = float(re.search(r"_U(\d+(?:\.\d+)?)", os.path.basename(os.path.normpath(d))).group(1))
        pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
        runs.append(dict(U=U, pfs=pfs, R=hugoniot(U)))
    runs.sort(key=lambda r: r["U"]); os.makedirs(os.path.join(outdir, "Contours"), exist_ok=True)
    col = [f"C{k}" for k in range(len(runs))]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True); meas = []
    for k, r in enumerate(runs):
        t, x, f = load(r["pfs"][-1]); R = r["R"]; n = len(x)
        i = np.where((f["rho"][n // 2:] - RHO0) > 0.5 * (R["a"] - 1) * RHO0)[0][-1] + n // 2
        xs = x[i] + (x[i + 1] - x[i]) * (f["rho"][i] - RHO0 * (1 + 0.5 * (R["a"] - 1))) / (f["rho"][i] - f["rho"][i + 1])
        m = (x > 0.25 * xs) & (x < 0.75 * xs)
        meas.append((r["U"], xs / t, f["rho"][m].mean() / RHO0, (f["p"] - f["S11"])[m].mean()))
        for a_, key, ref in ((ax[0, 0], "rho", R["a"] * RHO0), (ax[0, 1], "p", R["p"]), (ax[1, 0], "S11", R["S11"]), (ax[1, 1], "u", None)):
            a_.plot(x, f[key], "-", color=col[k], lw=1.1, label=f"Numerical, U = {r['U']:g}")
            if ref is not None: a_.plot([-R["s"] * t, R["s"] * t], [ref, ref], "--", color="k", lw=0.9, label="Exact" if k == 0 else None)
            for sg in (-1, 1): a_.axvline(sg * R["s"] * t, color=col[k], ls=":", lw=0.7)
    for a_, lab in zip(ax.ravel(), (r"$\rho$", r"$p$", r"$S_{11}$", r"$u$")): a_.set_ylabel(lab); a_.grid(alpha=0.3)
    for a_ in ax[1]: a_.set_xlabel("x")
    ax[0, 0].legend(fontsize=8); fig.suptitle(f"Uniaxial Shock in an Elastic Solid at t = {t:.3f}", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "piston_profiles.png"), dpi=200); plt.close(fig)

    meas = np.array(meas); UU = np.linspace(0.05, 1.15 * meas[:, 0].max(), 80); H = [hugoniot(u) for u in UU]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for a_, yex, j, lab in ((ax[0], [h["s"] for h in H], 1, "Shock Speed"), (ax[1], [h["a"] for h in H], 2, r"Compression $\rho_1/\rho_0$"),
                            (ax[2], [h["p"] - h["S11"] for h in H], 3, "Normal Stress")):
        a_.plot(UU, yex, "k-", lw=1.1, label="Exact"); a_.plot(meas[:, 0], meas[:, j], "o", color="C3", label="Numerical")
        a_.set_xlabel("Piston Speed U"); a_.set_ylabel(lab); a_.grid(alpha=0.3); a_.legend(fontsize=9)
    fig.suptitle("Uniaxial Shock in an Elastic Solid: Hugoniot", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "piston_hugoniot.png"), dpi=200); plt.close(fig)
    for U, sp, a, sg in meas:
        R = hugoniot(U); print(f"  U = {U:g}: shock speed {sp:.4f} (exact {R['s']:.4f}), compression {a:.5f} ({R['a']:.5f}), -sigma_11 {sg:.4f} ({R['p'] - R['S11']:.4f})")

    if make_gif:
        from PIL import Image
        nfr = min(len(r["pfs"]) for r in runs); frames = []
        gmax = [max(np.abs(np.gradient(load(r["pfs"][-1])[2]["rho"])).max(), 1e-30) for r in runs]
        for nf in range(nfr):
            fig = plt.figure(figsize=(9, 6.2)); gs = fig.add_gridspec(len(runs) + 1, 1, height_ratios=[0.45] * len(runs) + [3.2], hspace=0.12)
            axp = fig.add_subplot(gs[-1])
            for k, r in enumerate(runs):
                t, x, f = load(r["pfs"][nf]); a_ = fig.add_subplot(gs[k], sharex=axp)
                sch = np.exp(-6.0 * np.abs(np.gradient(f["rho"])) / gmax[k])               # 1 = uniform, dark = shock
                a_.imshow(sch[None, :], aspect="auto", extent=(-1, 1, 0, 1), cmap="gray", vmin=0, vmax=1)
                a_.set_yticks([]); a_.set_ylabel(f"U = {r['U']:g}", rotation=0, ha="right", va="center", fontsize=9); a_.tick_params(labelbottom=False)
                if k == 0: a_.set_title(f"Uniaxial Shock in an Elastic Solid, t = {t:.3f}    (strips: Numerical Schlieren)", fontsize=11)
                axp.plot(x, f["rho"], "-", color=col[k], lw=1.1, label=f"Numerical, U = {r['U']:g}")
                axp.plot([-r["R"]["s"] * t, r["R"]["s"] * t], [r["R"]["a"] * RHO0] * 2, "--", color="k", lw=0.9, label="Exact" if k == 0 else None)
            axp.set_xlim(-1, 1); axp.set_ylim(RHO0 - 0.02, max(r["R"]["a"] for r in runs) * RHO0 + 0.03); axp.set_xlabel("x"); axp.set_ylabel(r"$\rho$")
            axp.grid(alpha=0.3); axp.legend(fontsize=8, loc="upper right", ncol=2)
            fn = os.path.join(outdir, "Contours", f"frame_{nf:05d}.png"); fig.savefig(fn, dpi=100); plt.close(fig); frames.append(fn)
        ims = [Image.open(fn).convert("RGB") for fn in frames]; pal = ims[-1].quantize(colors=256, dither=Image.Dither.NONE)
        ims = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        ims[0].save(os.path.join(outdir, "Contours", "piston.gif"), save_all=True, append_images=ims[1:] + [ims[-1]] * 8, duration=110, loop=0)
    print(f"  wrote {outdir}/piston_profiles.png, piston_hugoniot.png" + (", Contours/piston.gif" if make_gif else ""))


if __name__ == "__main__":
    _a = [v for i, v in enumerate(sys.argv[1:]) if not v.startswith("--") and (i == 0 or sys.argv[i] != "--outdir")]
    _dirs = [v for v in _a if os.path.isdir(v)]
    if len(_dirs) > 1:
        _here = os.path.dirname(os.path.abspath(__file__))
        combined(_dirs, sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else os.path.join(_here, "Images", "Piston_combined"), "--no-gif" not in sys.argv)
    else:
        main()
