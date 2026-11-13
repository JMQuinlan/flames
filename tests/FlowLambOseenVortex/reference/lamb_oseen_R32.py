"""
Lamb-Oseen vortex: comparison of Hydro2 output with the exact viscous decay.

    u_theta(r,t) = Gamma/(2 pi r) * [1 - exp(-r^2 / (rc0^2 + 4 nu t))]

Replaces the KE-based nu_eff of lamb_oseen_single.py, which assumed an
exponential KE decay with wavenumber 1/rc0 (valid for Taylor-Green, not for
an isolated vortex, whose KE decays algebraically) and integrated the KE over
the whole periodic box, where the free-space solution does not hold.

Here, per plotfile:
  * azimuthal velocity of every cell inside r < R_EVAL about the vortex centre
    is compared with the exact value at the cell centre (L_inf and L2 errors
    normalised by the exact peak velocity);
  * a two-parameter least-squares fit u = G/(2 pi r) (1 - exp(-r^2/a)) gives
    the circulation G and core parameter a(t) = rc^2 + 4 nu t;
  * nu_eff = slope(a, t)/4  (linear regression over all frames);
  * peak velocity, core radius r_max = 1.12091 sqrt(a) and the kinetic energy
    inside the disc r < R_EVAL are compared with the exact values.

Usage:  python3 lamb_oseen_R32.py OUTDIR label1:plotdir1 [label2:plotdir2 ...]
Each plotdir holds the AMReX *cell plotfiles of one run.  The first run is
the one used for the profile figure; all runs enter the convergence table.
"""
import sys, os, glob, re, json
import numpy as np
import yt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

yt.funcs.mylog.setLevel(40)

GAMMA_C = 1.0      # circulation
RC0 = 0.5          # initial core radius
NU = 0.01          # mu/rho
XC, YC = 5.0, 5.0
R_EVAL = 3.0       # evaluation disc (box half-width is 5)
S_PEAK = 1.1209178 # r_max / sqrt(a)
F_PEAK = 0.6381726 # max_s (1-exp(-s^2))/s

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 15,
                     "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13})


def a_exact(t):
    return RC0**2 + 4.0 * NU * t


def uth_exact(r, t, G=GAMMA_C):
    a = a_exact(t)
    return G / (2 * np.pi * np.maximum(r, 1e-14)) * (1.0 - np.exp(-r**2 / a))


def model(r, G, a):
    return G / (2 * np.pi * np.maximum(r, 1e-14)) * (1.0 - np.exp(-r**2 / a))


def plotfiles(d):
    fs = [f for f in glob.glob(os.path.join(d, "*cell")) if ".old" not in f]
    return sorted(fs, key=lambda f: int(re.search(r"(\d+)cell$", f).group(1)))


def load(f):
    ds = yt.load(f)
    cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    x = np.array(cg["index", "x"])[:, :, 0]
    y = np.array(cg["index", "y"])[:, :, 0]
    g = lambda n: np.array(cg["boxlib", n])[:, :, 0]
    dx = float(ds.domain_width[0] / ds.domain_dimensions[0])
    return float(ds.current_time), x, y, g("velocityx"), g("velocityy"), g("density"), g("pressure"), dx


def analyse(d):
    rows = []
    prof = {}
    seen = set()
    for f in plotfiles(d):
        t, x, y, u, v, rho, p, dx = load(f)
        if round(t, 3) in seen:
            continue
        seen.add(round(t, 3))
        # vortex centre: vorticity-weighted centroid inside the evaluation disc
        w = (np.gradient(v, dx, axis=0) - np.gradient(u, dx, axis=1))
        rr0 = np.hypot(x - XC, y - YC)
        m0 = (rr0 < 1.5) & (w > 0)
        xc = np.sum(w[m0] * x[m0]) / np.sum(w[m0]); yc = np.sum(w[m0] * y[m0]) / np.sum(w[m0])
        dxr, dyr = x - XC, y - YC          # compare about the nominal centre
        r = np.hypot(dxr, dyr)
        ut = (-u * dyr + v * dxr) / np.maximum(r, 1e-14)
        ur = (u * dxr + v * dyr) / np.maximum(r, 1e-14)
        m = (r < R_EVAL) & (r > 0.5 * dx)
        ue = uth_exact(r, t)
        upk = GAMMA_C / (2 * np.pi * np.sqrt(a_exact(t))) * F_PEAK
        err = ut[m] - ue[m]
        linf = np.max(np.abs(err)) / upk
        l2 = np.sqrt(np.mean(err**2)) / upk
        (Gf, af), _ = curve_fit(model, r[m], ut[m], p0=[GAMMA_C, a_exact(t)])
        upk_num = np.max(ut[m])
        rpk_num = S_PEAK * np.sqrt(af)
        ke_num = 0.5 * np.sum(rho[m] * (u[m]**2 + v[m]**2)) * dx * dx
        ke_ex = 0.5 * np.sum(1.0 * ue[m]**2) * dx * dx
        drho = np.max(np.abs(rho[r < R_EVAL] - 1.0))
        rows.append(dict(t=t, linf=linf, l2=l2, G=Gf, a=af, a_ex=a_exact(t),
                         upk=upk_num, upk_ex=upk, rpk=rpk_num, rpk_ex=S_PEAK * np.sqrt(a_exact(t)),
                         ke=ke_num, ke_ex=ke_ex, urmax=np.max(np.abs(ur[m])) / upk,
                         drho=drho, xc=xc, yc=yc, dx=dx))
        prof[t] = (r[m].ravel(), ut[m].ravel(), np.hypot(x - xc, y - yc)[m].ravel())
    tt = np.array([q["t"] for q in rows]); aa = np.array([q["a"] for q in rows])
    slope, icpt = np.polyfit(tt, aa, 1)
    return rows, prof, slope / 4.0, icpt


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    runs = []
    for spec in sys.argv[2:]:
        lab, d = spec.split(":", 1)
        rows, prof, nu_eff, icpt = analyse(d)
        runs.append((lab, rows, prof, nu_eff, icpt))
        print(f"\n=== {lab}  ({d})  dx = {rows[0]['dx']:.5g}")
        print(" t      Linf%    L2%    upk_err%  rpk_err%  KE_err%   G_fit    ur_max%  max|drho|  centre")
        for q in rows:
            print(f"{q['t']:5.2f} {100*q['linf']:8.4f} {100*q['l2']:7.4f} {100*(q['upk']/q['upk_ex']-1):9.4f} "
                  f"{100*(q['rpk']/q['rpk_ex']-1):9.4f} {100*(q['ke']/q['ke_ex']-1):8.4f} {q['G']:8.5f} "
                  f"{100*q['urmax']:8.4f} {q['drho']:9.2e}  ({q['xc']:.4f},{q['yc']:.4f})")
        print(f" nu_eff = {nu_eff:.6e}  (exact {NU})  error = {100*(nu_eff/NU-1):.3f} %;  fitted rc0^2 = {icpt:.5f} (exact {RC0**2})")
    summary = []
    for lab, rows, prof, nu_eff, icpt in runs:
        late = [q for q in rows if q["t"] > 0]
        summary.append(dict(label=lab, dx=rows[0]["dx"],
                            linf=max(q["linf"] for q in late), l2=max(q["l2"] for q in late),
                            upk=max(abs(q["upk"] / q["upk_ex"] - 1) for q in late),
                            rpk=max(abs(q["rpk"] / q["rpk_ex"] - 1) for q in late),
                            ke=max(abs(q["ke"] / q["ke_ex"] - 1) for q in late),
                            G=max(abs(q["G"] - 1) for q in late),
                            nu=nu_eff / NU - 1, t_end=rows[-1]["t"]))
    print("\n=== Summary (max over t > 0 of |error|, %):")
    print(" run                 dx       Linf    L2     upk    rpk    KE     Gamma   nu_eff")
    for s in summary:
        print(f" {s['label']:18s} {s['dx']:.5f} {100*s['linf']:6.3f} {100*s['l2']:6.3f} {100*s['upk']:6.3f} "
              f"{100*s['rpk']:6.3f} {100*s['ke']:6.3f} {100*s['G']:6.3f} {100*s['nu']:7.3f}   t_end={s['t_end']}")
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(dict(summary=summary, runs={lab: rows for lab, rows, *_ in runs}), fh, indent=1)

    # ---------- figures (first run = primary) ----------
    lab0, rows0, prof0, nu0, _ = runs[0]
    times = sorted(prof0.keys())
    pick = [times[int(np.argmin([abs(t - s) for t in times]))] for s in (0.0, 2.5, 5.0, 10.0)]
    mk = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    rl = np.linspace(1e-3, R_EVAL, 400)
    for i, t in enumerate(pick):
        r, ut, _ = prof0[t]
        # bin-average along r for markers (one per cell width)
        dx = rows0[0]["dx"]
        bins = np.arange(0, R_EVAL + dx, 2 * dx)
        idx = np.digitize(r, bins)
        rb = np.array([r[idx == k].mean() for k in np.unique(idx)])
        ub = np.array([ut[idx == k].mean() for k in np.unique(idx)])
        ax.plot(rl, uth_exact(rl, t), "k-", lw=1.8, label="Exact" if i == 0 else None)
        ax.plot(rb, ub, mk[i % 4], mfc="none", color="tab:red", ms=7, markevery=max(1, len(rb) // 25),
                label=f"Numerical, $t = {t:.1f}$")
    ax.set_xlabel("Radius, $r$"); ax.set_ylabel(r"Azimuthal Velocity, $u_\theta$")
    emax = max(q["linf"] for q in rows0 if q["t"] > 0)
    ax.set_title("Lamb-Oseen Vortex Azimuthal Velocity Profile")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    for e in ("png", "eps"):
        fig.savefig(os.path.join(out, f"LambOseen_Profile.{e}"), dpi=300)
    plt.close(fig)

    # peak velocity and core radius vs time
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    tl = np.linspace(0, times[-1], 200)
    tt = np.array([q["t"] for q in rows0])
    axs[0].plot(tl, GAMMA_C * F_PEAK / (2 * np.pi * np.sqrt(a_exact(tl))), "k-", lw=1.8, label="Exact")
    axs[0].plot(tt, [q["upk"] for q in rows0], "o", mfc="none", color="tab:red", ms=7, label="Numerical")
    axs[0].set_xlabel("Time, $t$"); axs[0].set_ylabel(r"Peak Azimuthal Velocity, $u_{\theta,\max}$")
    axs[0].set_title("Peak Velocity Decay")
    axs[1].plot(tl, S_PEAK * np.sqrt(a_exact(tl)), "k-", lw=1.8, label="Exact")
    axs[1].plot(tt, [q["rpk"] for q in rows0], "o", mfc="none", color="tab:red", ms=7, label="Numerical")
    axs[1].set_xlabel("Time, $t$"); axs[1].set_ylabel(r"Core Radius, $r_{\max}$")
    axs[1].set_title("Core Radius Growth")
    for a_ in axs:
        a_.grid(True, alpha=0.3); a_.legend()
    fig.tight_layout()
    for e in ("png", "eps"):
        fig.savefig(os.path.join(out, f"LambOseen_Peak_Core.{e}"), dpi=300)
    plt.close(fig)

    # KE inside disc + core parameter a(t)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    axs[0].plot(tt, [q["ke_ex"] for q in rows0], "k-", lw=1.8, label="Exact")
    axs[0].plot(tt, [q["ke"] for q in rows0], "o", mfc="none", color="tab:red", ms=7, label="Numerical")
    axs[0].set_xlabel("Time, $t$"); axs[0].set_ylabel(f"Kinetic Energy, $r < {R_EVAL:g}$")
    axs[0].set_title("Kinetic Energy Decay")
    axs[1].plot(tl, a_exact(tl), "k-", lw=1.8, label="Exact")
    axs[1].plot(tt, [q["a"] for q in rows0], "o", mfc="none", color="tab:red", ms=7, label="Numerical")
    axs[1].set_xlabel("Time, $t$"); axs[1].set_ylabel(r"Core Parameter, $r_c^2 + 4\nu t$")
    axs[1].set_title(r"Core Spreading, $\nu_{\mathrm{eff}}/\nu = %.4f$" % (nu0 / NU))
    for a_ in axs:
        a_.grid(True, alpha=0.3); a_.legend()
    fig.tight_layout()
    for e in ("png", "eps"):
        fig.savefig(os.path.join(out, f"LambOseen_KE_Core.{e}"), dpi=300)
    plt.close(fig)

    # error vs time for all runs
    fig, ax = plt.subplots(figsize=(8, 5.2))
    mks = ["o", "s", "^", "D", "v"]
    for i, (lab, rows, *_r) in enumerate(runs):
        q = [x for x in rows if x["t"] > 0]
        ax.semilogy([x["t"] for x in q], [100 * x["linf"] for x in q], mks[i % 5] + "-", mfc="none", lw=1, ms=7,
                    label=lab)
    ax.set_xlabel("Time, $t$"); ax.set_ylabel(r"Max $|u_\theta - u_\theta^{\mathrm{exact}}| / u_{\theta,\max}$ [%]")
    ax.set_title("Lamb-Oseen Vortex Velocity Error")
    ymax = max(100 * x["linf"] for _l, rows, *_r in runs for x in rows)
    ax.set_ylim(top=ymax * 30)
    ax.grid(True, which="both", alpha=0.3); ax.legend(ncol=2, fontsize=12, loc="upper center"); fig.tight_layout()
    for e in ("png", "eps"):
        fig.savefig(os.path.join(out, f"LambOseen_Error_Time.{e}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
