"""
Rayleigh-Taylor unit test: Hydro2 vs. (i) linear theory, (ii) Goncharov's
nonlinear bubble model, (iii) the low-diffusion HLLC/MUSCL reference
(rt_reference_hllc.py) on a sequence of grids.

  * interface = eta = 0.5 (Hydro2) / phi = 0.5 (reference colour function);
  * bubble tip  y_b: highest 0.5-crossing over all columns (bubble at x = lambda/2);
    spike tip   y_s: lowest 0.5-crossing over all columns (spike at x = 0 = 0.5);
  * amplitude  h = (y_b - y_s)/2, linear growth rate from a least-squares fit
    of h = a exp(sigma t) + b exp(-sigma t) over h < H_HI (k h < 0.4);
  * linear theory (Chandrasekhar 1961, inviscid, immiscible, incompressible,
    walls at kh = 9.4 so coth(kh) = 1):  sigma = sqrt(A g k);
  * Goncharov (2002) asymptotic bubble velocity (2D, C = 3):
        U_b = sqrt( 2 A g / ((1 + A) C k) ).

Usage:
  python3 rt_compare_R32.py OUTDIR  hydro:LABEL:PLTDIR ...  ref:LABEL:FILE.npz ...
"""
import sys, os, glob, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LX, LY, Y0 = 0.5, 1.5, 0.75
G = 0.1
A = (2.0 - 1.0) / (2.0 + 1.0)
K = 2 * np.pi / LX
SIGMA = np.sqrt(A * G * K)
UB_GON = np.sqrt(2 * A * G / ((1 + A) * 3.0 * K))
H_LO, H_HI = 0.004, 0.03
TMAX_PLOT = 15.0

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 15,
                     "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13})


def crossings(col, y, level=0.5):
    s = col - level
    idx = np.where(np.sign(s[:-1]) != np.sign(s[1:]))[0]
    return y[idx] + (y[idx + 1] - y[idx]) * (-s[idx]) / (s[idx + 1] - s[idx])


def tips(phi, y):
    """Mixing-zone edges: bubble tip = highest 0.5-crossing over all columns,
    spike tip = lowest 0.5-crossing over all columns (robust to roll-ups at the
    symmetry lines, where the column crossings become multi-valued)."""
    yb, ys = -np.inf, np.inf
    for i in range(phi.shape[0]):
        c = crossings(phi[i, :], y)
        if len(c):
            yb = max(yb, c.max()); ys = min(ys, c.min())
    return (yb if np.isfinite(yb) else np.nan), (ys if np.isfinite(ys) else np.nan)


def mode_amp(phi, x, y):
    """Fourier cos(kx) amplitude of the equivalent interface height
    y_i(x) = L_y - int phi dy (phi = 1 heavy above).  The column integral is
    exact for any interface-profile shape, so the measure is insensitive to the
    numerical broadening of the diffuse/smeared interface; valid while the
    interface is single-valued (linear regime)."""
    dy = y[1] - y[0]
    yi = LY - phi.sum(axis=1) * dy
    return abs(2.0 * np.mean((yi - yi.mean()) * np.cos(K * x)))


def growth_fit(t, a, amax):
    """sigma from a(t) = c1 exp(sigma t) + c2 exp(-sigma t) over 0 < t, a < amax."""
    from scipy.optimize import curve_fit
    m = (a < amax) & (t > 0) & np.isfinite(a)
    try:
        p, _ = curve_fit(lambda tt, c1, c2, s: c1 * np.exp(s * tt) + c2 * np.exp(-s * tt), t[m], a[m],
                         p0=[a[m][-1] * np.exp(-SIGMA * t[m][-1]), -a[m][-1] * np.exp(-SIGMA * t[m][-1]), SIGMA],
                         maxfev=20000)
        return p[2], m.sum()
    except Exception:
        return np.nan, m.sum()


def interface_width(phi, x, y):
    """Mean 5-95 % interface thickness in cells: area of cells with
    0.05 < phi < 0.95 divided by the length of the phi = 0.5 contour, over dx."""
    import matplotlib.pyplot as _plt
    dx = x[1] - x[0]
    f = _plt.figure(); cs = _plt.contour(x, y, phi.T, levels=[0.5]); _plt.close(f)
    L = 0.0
    for seg in cs.allsegs[0]:
        L += np.sum(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1])))
    n = np.sum((phi > 0.05) & (phi < 0.95))
    return n * dx * dx / (L * dx) if L > 0 else np.nan


def load_hydro(d):
    import yt
    yt.funcs.mylog.setLevel(40)
    fs = [f for f in glob.glob(os.path.join(d, "*cell")) if ".old" not in f]
    fs = sorted(fs, key=lambda f: int(re.search(r"(\d+)cell$", f).group(1)))
    out = dict(t=[], phi=[], rho=[], vort=[], u=[], v=[])
    seen = set()
    for f in fs:
        ds = yt.load(f)
        t = round(float(ds.current_time), 6)
        if t in seen:
            continue
        seen.add(t)
        cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        g = lambda n: np.array(cg["boxlib", n])[:, :, 0]
        out["t"].append(float(ds.current_time))
        out["phi"].append(g("eta")); out["rho"].append(g("density"))
        out["u"].append(g("velocityx")); out["v"].append(g("velocityy"))
        nx, ny = ds.domain_dimensions[:2]
    dx = LX / nx
    out["x"] = (np.arange(nx) + 0.5) * dx; out["y"] = (np.arange(ny) + 0.5) * dx
    for k in ("t",):
        out[k] = np.array(out[k])
    return out


def load_ref(f):
    z = np.load(f)
    return dict(t=z["t"], phi=z["phi"], rho=z["rho"], u=z["u"], v=z["v"], x=z["x"], y=z["y"])


def series(D):
    yb, ys = [], []
    for ph in D["phi"]:
        b, s = tips(np.asarray(ph, float), D["y"]); yb.append(b); ys.append(s)
    yb = np.array(yb); ys = np.array(ys)
    h = 0.5 * (yb - ys)
    t = np.asarray(D["t"])
    # linear phase: h < H_HI (k h < 0.38).  The deck's velocity seed is not the
    # RT eigenmode, so h(t) = a exp(sigma t) + b exp(-sigma t) (growing +
    # decaying linear modes) is fitted with sigma free.
    from scipy.optimize import curve_fit
    m = (h < H_HI) & (t > 0) & np.isfinite(h)
    try:
        p, _ = curve_fit(lambda tt, a, b, s: a * np.exp(s * tt) + b * np.exp(-s * tt), t[m], h[m],
                         p0=[0.004, -0.004, SIGMA])
        sig = p[2]
    except Exception:
        sig = np.nan
    return t, yb, ys, h, sig, m.sum()


def restrict(f, r):
    nx, ny = f.shape
    return f.reshape(nx // r, r, ny // r, r).mean(axis=(1, 3))


def vorticity(u, v, dx):
    return np.gradient(v, dx, axis=0) - np.gradient(u, dx, axis=1)


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    runs = []
    for spec in sys.argv[2:]:
        kind, lab, path = spec.split(":", 2)
        D = load_hydro(path) if kind == "hydro" else load_ref(path)
        D["kind"] = kind; D["label"] = lab
        D["N"] = len(D["x"])
        D["series"] = series(D)
        runs.append(D)
    print(f"A = {A:.4f}, k = {K:.4f}, sigma_lin = {SIGMA:.5f} 1/s, U_b(Goncharov) = {UB_GON:.5f} m/s")
    res = {}
    print(" run                 N     sigma_fit   err%    npts   y_b(t)/y_s(t) at t=5,10,15,20")
    for D in runs:
        t, yb, ys, h, sig, npt = D["series"]
        def at(T, a):
            i = np.argmin(abs(t - T)); return a[i] if abs(t[i] - T) < 1e-3 else np.nan
        s = " ".join(f"{at(T, yb):.4f}/{at(T, ys):.4f}" for T in (5, 10, 15, 20))
        print(f" {D['label']:18s} {D['N']:4d}  {sig:9.5f}  {100*(sig/SIGMA-1):7.2f}  {npt:4d}   {s}")
        # bubble velocity (central difference), report the plateau mean over h in [0.1, 0.4] lambda
        vb = np.gradient(yb, t)
        mp = (t >= 6.0) & (t <= 12.0)
        print(f"     bubble velocity mean over 6<=t<=12: {vb[mp].mean():.5f} m/s ({100*(vb[mp].mean()/UB_GON-1):+.1f} % vs Goncharov), "
              f"min {vb[mp].min():.5f} max {vb[mp].max():.5f}")
        res[D["label"]] = dict(kind=D["kind"], N=D["N"], sigma=sig, t=t.tolist(), yb=yb.tolist(), ys=ys.tolist(),
                               vb=vb.tolist())
    # finest reference
    refs = sorted([D for D in runs if D["kind"] == "ref"], key=lambda D: D["N"])
    best = refs[-1] if refs else None
    if best is not None:
        tb, ybb, ysb, *_ = best["series"]
        print(f"\nTip-position differences vs finest reference ({best['label']}), % of wavelength (lambda = {LX}):")
        for D in runs:
            if D is best:
                continue
            t, yb, ys, *_ = D["series"]
            rows = []
            for T in (5, 7.5, 10, 12.5, 15, 17.5, 20):
                i = np.argmin(abs(t - T)); j = np.argmin(abs(tb - T))
                if abs(t[i] - T) < 1e-3 and abs(tb[j] - T) < 1e-3:
                    rows.append((T, 100 * (yb[i] - ybb[j]) / LX, 100 * (ys[i] - ysb[j]) / LX))
            print(f"  {D['label']:18s} " + "  ".join(f"t={T:g}: {b:+.2f}/{s:+.2f}" for T, b, s in rows))
            res[D["label"]]["tipdiff_vs_best"] = rows
        # field L1 differences of the colour function on each Hydro2 grid (reference restricted)
        print("\nL1 colour-function difference  sum|phi_H - phi_ref| dx^2 / lambda  (= mean interface offset, m):")
        for D in runs:
            if D is best:
                continue
            r = best["N"] // D["N"]
            if r < 1 or best["N"] % D["N"]:
                continue
            rows = []
            for T in (5, 10, 12.5, 15, 20):
                i = np.argmin(abs(np.asarray(D["t"]) - T)); j = np.argmin(abs(tb - T))
                if abs(D["t"][i] - T) < 1e-3 and abs(tb[j] - T) < 1e-3:
                    pr = restrict(np.asarray(best["phi"][j], float), r) if r > 1 else np.asarray(best["phi"][j], float)
                    dx = LX / D["N"]
                    rows.append((T, np.sum(np.abs(np.asarray(D["phi"][i], float) - pr)) * dx * dx / LX))
            print(f"  {D['label']:18s} " + "  ".join(f"t={T:g}: {e:.4e}" for T, e in rows))
            res[D["label"]]["L1phi_vs_best"] = rows
    with open(os.path.join(out, "rt_summary.json"), "w") as fh:
        json.dump(dict(sigma_lin=SIGMA, Ub_goncharov=UB_GON, runs=res), fh, indent=1)

    # ---------------- figures ----------------
    mks = ["o", "s", "^", "D", "v", "<", ">"]
    hyd = [D for D in runs if D["kind"] == "hydro"]
    # (linear growth rate: see rt_linear_R32.py -- the deck's seed has no clean exponential window)
    # 2. bubble & spike tip positions vs reference
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for D in refs:
        t, yb, ys, *_ = D["series"]
        st = "-" if D is best else (":" if "Rusanov" in D["label"] else "--")
        col = "k" if D is best else ("tab:green" if "Rusanov" in D["label"] else "0.55")
        ax.plot(t, yb, st, color=col, lw=1.8 if D is best else 1.2, label=D["label"])
        ax.plot(t, ys, st, color=col, lw=1.8 if D is best else 1.2)
    for i, D in enumerate(hyd):
        t, yb, ys, *_ = D["series"]
        ax.plot(t, yb, mks[i % 7], mfc="none", color="tab:red", ms=7, label=f"Numerical, {D['N']}$\\times${3*D['N']}")
        ax.plot(t, ys, mks[i % 7], mfc="none", color="tab:red", ms=7)
    ax.axhline(Y0, color="0.7", lw=0.8)
    ax.text(1.0, 0.82, "Bubble Tip", fontsize=13); ax.text(1.0, 0.66, "Spike Tip", fontsize=13)
    ax.set_xlabel("Time, $t$ [s]"); ax.set_ylabel("Tip Position, $y$ [m]")
    ax.set_title("Rayleigh-Taylor Bubble and Spike Tips"); ax.set_xlim(0, TMAX_PLOT)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=11, frameon=False)
    fig.set_size_inches(8.5, 7.8); fig.tight_layout()
    for e in ("png", "eps"):
        fig.savefig(os.path.join(out, f"RT_Tip_Positions.{e}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. bubble velocity vs Goncharov
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for D in refs:
        t, yb, *_ = D["series"]
        ax.plot(t, np.gradient(yb, t), "-" if D is best else (":" if "Rusanov" in D["label"] else "--"),
                color="k" if D is best else ("tab:green" if "Rusanov" in D["label"] else "0.55"),
                lw=1.6 if D is best else 1.1, label=D["label"])
    for i, D in enumerate(hyd):
        t, yb, *_ = D["series"]
        ax.plot(t, np.gradient(yb, t), mks[i % 7], mfc="none", color="tab:red", ms=7,
                label=f"Numerical, {D['N']}$\\times${3*D['N']}")
    ax.axhline(UB_GON, color="tab:blue", lw=1.8, ls="-.", label="Goncharov (2002)")
    ax.set_xlabel("Time, $t$ [s]"); ax.set_ylabel("Bubble Velocity, $\\dot{y}_b$ [m/s]")
    ax.set_title("Rayleigh-Taylor Bubble Velocity"); ax.set_xlim(0, TMAX_PLOT); ax.set_ylim(-0.005, 0.07)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=11, frameon=False)
    fig.set_size_inches(8.5, 7.8); fig.tight_layout()
    for e in ("png", "eps"):
        fig.savefig(os.path.join(out, f"RT_Bubble_Velocity.{e}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3b. interface thickness (5-95 % of phi / eta) in cells
    fig, ax = plt.subplots(figsize=(8, 5.2))
    print("\n5-95 % interface thickness [cells]  (t = 0, 2.5, 5, 7.5, 10, 12.5):")
    ih = 0
    for D in runs:
        t = np.asarray(D["t"]); sel = t <= 12.5 + 1e-6
        w = np.array([interface_width(np.asarray(D["phi"][i], float), D["x"], D["y"]) for i in np.where(sel)[0]])
        D["width"] = (t[sel], w)
        row = []
        for T in (0, 2.5, 5, 7.5, 10, 12.5):
            j = np.argmin(abs(t[sel] - T)); row.append(f"{w[j]:.2f}")
        print(f"  {D['label']:40s} " + "  ".join(row) + f"   mean(2.5<=t<=12.5) = {w[t[sel] >= 2.5].mean():.2f}")
        if D["kind"] == "hydro":
            ax.plot(t[sel], w, mks[ih % 7], mfc="none", color="tab:red", ms=7, label=f"Numerical, {D['N']}$\\times${3*D['N']}"); ih += 1
        else:
            st = ":" if "Rusanov" in D["label"] else ("-" if "THINC" in D["label"] else "--")
            ax.plot(t[sel], w, st, lw=1.6, label=D["label"])
    ax.set_xlabel("Time, $t$ [s]"); ax.set_ylabel("Interface Thickness (5-95 %) [cells]")
    ax.set_title("Rayleigh-Taylor Interface Thickness")
    ax.grid(True, alpha=0.3); ax.set_ylim(0, None)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=11, frameon=False)
    fig.set_size_inches(8.5, 7.4); fig.tight_layout()
    for e in ("png", "eps"):
        fig.savefig(os.path.join(out, f"RT_Interface_Width.{e}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. side-by-side density at selected times: reference (left half) | Hydro2 (right half)
    if best is not None and hyd:
        H = sorted(hyd, key=lambda D: D["N"])
        for T in (10.0, 12.5, 14.5, 15.0, 17.5, 20.0):
            j = np.argmin(abs(best["t"] - T))
            if abs(best["t"][j] - T) > 1e-3:
                continue
            panels = []
            for D in refs:
                if "Rusanov" in D["label"] and D["N"] == min(q["N"] for q in refs):
                    i = np.argmin(abs(np.asarray(D["t"]) - T))
                    if abs(D["t"][i] - T) < 1e-3:
                        panels.append((D["label"].split(",")[0].replace("Original Reference", "Original"), D, i))
            panels.append((best["label"].split(",")[0], best, j))
            for D in H:
                i = np.argmin(abs(np.asarray(D["t"]) - T))
                if abs(D["t"][i] - T) < 1e-3:
                    panels.append(("Numerical", D, i))
            fig, axs = plt.subplots(1, len(panels), figsize=(2.5 * len(panels) + 1.6, 6.2), sharey=True,
                                    gridspec_kw=dict(wspace=0.3))
            axs = np.atleast_1d(axs)
            for a_, (nm, D, i) in zip(axs, panels):
                im = a_.imshow(np.asarray(D["rho"][i], float).T, origin="lower", extent=[0, LX, 0, LY],
                               cmap="viridis", vmin=0.9, vmax=2.1, interpolation="nearest")
                a_.contour(D["x"], D["y"], np.asarray(D["phi"][i], float).T, levels=[0.5], colors="w", linewidths=0.8)
                a_.set_title(nm.replace(" (", "\n(") + f"\n{D['N']}$\\times${3*D['N']}", fontsize=12)
                a_.set_xlabel("$x$ [m]"); a_.set_xticks([0.0, 0.25, 0.5]); a_.set_xticklabels(["0", "0.25", "0.5"])
            axs[0].set_ylabel("$y$ [m]")
            cb = fig.colorbar(im, ax=list(axs), shrink=0.85); cb.set_label(r"Density, $\rho$ [kg/m$^3$]")
            fig.suptitle(f"Rayleigh-Taylor Density, $t = {T:g}$ s", fontsize=15)
            for e in ("png", "eps"):
                fig.savefig(os.path.join(out, f"RT_Density_t{T:05.1f}.{e}"), dpi=250, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
