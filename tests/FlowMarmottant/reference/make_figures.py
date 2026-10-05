"""Build every figure and the results table for the Marmottant paper.

Inputs are the CSV/NPZ files written by extract_laplace.py, extract_fields.py
and operator_test.py into ../data.  Outputs go to ../figures and ../tables.
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

CHI, RB, SBRK, SIGW = 0.55, 1.0e-3, 0.073, 0.073
EPS = 7.8125e-5
P_INF = 101325.0
TW = 6.5  # text width, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.3,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})
C_BUCK, C_ELAS, C_RUPT = "#0072B2", "#009E73", "#D55E00"
C_INK, C_MUTED = "#222222", "#777777"
LS = {"buckled": ":", "elastic": "-", "ruptured": "--"}
REG_C = {"buckled": C_BUCK, "elastic": C_ELAS, "ruptured": C_RUPT}


def sigma_R(R):
    R = np.asarray(R, float)
    el = CHI * (R**2 / RB**2 - 1.0)
    return np.where(R <= RB, 0.0, np.where(el >= SBRK, SIGW, el))


def sigma_G(G, Gb):
    if G <= 0 or G >= Gb:
        return 0.0
    el = CHI * (Gb / G - 1.0)
    return SIGW if el >= SBRK else el


def regime(R0):
    s = CHI * (R0**2 / RB**2 - 1.0)
    return "buckled" if R0 <= RB else ("ruptured" if s >= SBRK else "elastic")


def load(tag=""):
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, f"laplace_{tag}R*.csv"))):
        R0 = float(open(f).readline().split("R0=")[1].split()[0])
        d = np.genfromtxt(f, delimiter=",", names=True, skip_header=1)
        out[R0] = d
    return out


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    fig.savefig(os.path.join(FIG, name + ".eps"))
    plt.close(fig)
    print("wrote", name)


def cmap_R0(keys):
    cm = plt.get_cmap("viridis")
    k = sorted(keys)
    return {R0: cm(0.05 + 0.85 * i / max(len(k) - 1, 1)) for i, R0 in enumerate(k)}



# ---------------------------------------------------------------------------
# Single-panel figures (one message per figure). W x H in inches.
W1, H1 = 5.6, 3.3
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                     "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9})
LREG = {"buckled": "buckled ($R_0\\leq R_{\\mathrm{b}}$)", "elastic": "elastic",
        "ruptured": "ruptured ($\\sigma=\\sigma_w$)"}


def one(w=W1, h=H1):
    return plt.subplots(figsize=(w, h))


def r0_legend(ax, runs, col, keys=None, loc=None, ncol=4, title=r"Initial radius $R_0$ (line style: regime)"):
    keys = sorted(runs) if keys is None else keys
    hs = [ax.plot([], [], LS[regime(R0)], color=col[R0], lw=1.4, label=f"{R0*1e3:.3f} mm, {regime(R0)}")[0]
          for R0 in keys]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol,
              title=title, title_fontsize=8.5, fontsize=7.6, columnspacing=1.0, handlelength=2.2)


def fig_law_R(runs):
    fig, a = one(5.6, 3.2)
    R = np.linspace(0.8e-3, 1.45e-3, 800)
    Rr = RB * np.sqrt(1.0 + SBRK / CHI)
    a.axvspan(0.8, 1.0, color=C_BUCK, alpha=0.08, lw=0)
    a.axvspan(1.0, Rr / RB, color=C_ELAS, alpha=0.10, lw=0)
    a.axvspan(Rr / RB, 1.45, color=C_RUPT, alpha=0.08, lw=0)
    a.plot(R / RB, 1e3 * sigma_R(R), color=C_INK, label=r"$\sigma(R)$, Marmottant et al. (2005)")
    for i, R0 in enumerate(sorted(runs)):
        a.plot(R0 / RB, 1e3 * sigma_R(R0), "o", ms=6, mfc=REG_C[regime(R0)], mec="white", mew=0.8, zorder=5,
               label="Laplace sweep" if i == 0 else None)
    a.text(0.9,   75, "Buckled", ha="center", color=C_BUCK)
    a.text(1.035, 75, "Elastic", ha="center", color=C_ELAS)
    a.text(1.26,  75, "Ruptured", ha="center", color=C_RUPT)
    a.set_xlabel(r"Normalized bubble radius, $R/R_{\mathrm{b}}$")
    a.set_ylabel(r"Surface Tension, $\sigma$ (mN m$^{-1}$)")
    a.set_xlim(0.8, 1.45)
    a.set_ylim(-3, 80)
    a.legend(loc="lower right")
    save(fig, "fig_law_R")


def fig_law_kin():
    fig, a = one(5.6, 3.2)
    R0 = 1.025e-3
    Gb = (R0 / RB) ** 2
    R = np.linspace(0.94e-3, 1.10e-3, 800)
    a.plot(R / R0, 1e3 * sigma_R(R), color=C_INK, lw=3.0, alpha=0.25, label=r"Marmottant law $\sigma(R)$")
    a.plot(R / R0, 1e3 * np.array([sigma_G((R0 / r) ** 2, Gb) for r in R]), color=C_ELAS,
           label=r"$\sigma_{\mathrm{eff}}(\Gamma)$, spherical: $\Gamma=(R_0/R)^2$")
    a.plot(R / R0, 1e3 * np.array([sigma_G(R0 / r, Gb) for r in R]), color=C_BUCK, ls="--",
           label=r"$\sigma_{\mathrm{eff}}(\Gamma)$, cylindrical: $\Gamma=R_0/R$")
    a.axvline(1.0, color=C_MUTED, lw=0.7, ls=":")
    a.set_xlabel(r"Bubble radius normalised by reference radius, $R/R_0$")
    a.set_ylabel(r"Effective surface tension, $\sigma_{\mathrm{eff}}$ (mN m$^{-1}$)")
    a.legend(loc="upper left")
    save(fig, "fig_law_kin")


def fig_operator():
    d = np.load(os.path.join(DATA, "operator_test.npz"))
    R0, eps, sig = float(d["R0"]), float(d["eps"]), float(d["sigma"])
    ratios = d["ratios"]
    scale = sig / (R0 * eps)
    # force profile
    fig, a = one(5.6, 3.2)
    xr = d["xr_exact_16"]
    a.plot((xr - R0) / eps, d["Fr_exact_16"] / scale, color=C_INK, lw=3.0, alpha=0.3, label="analytic, $-\\sigma c'(r)/r$")
    cm = plt.get_cmap("viridis")
    for rt, mk, cc in ((1, "s", cm(0.1)), (2, "o", cm(0.45)), (4, "^", cm(0.8))):
        x = d[f"xr_css_{rt}"]
        m = np.abs(x - R0) < 4 * eps
        a.plot((x[m] - R0) / eps, d[f"Fr_css_{rt}"][m] / scale, mk + "-", ms=4, lw=0.9, color=cc,
               label=rf"discrete $\nabla\cdot\Omega$, $\varepsilon/\Delta x={rt}$")
    a.set_xlim(-4, 4)
    a.set_xlabel(r"Distance from interface in band widths, $(r-R_0)/\varepsilon$")
    a.set_ylabel(r"Radial capillary force, $F_r R_0\varepsilon/\sigma$")
    a.legend(loc="lower left")
    save(fig, "fig_operator_force")
    # balance
    fig, a = one(5.0, 3.2)
    xc = np.array([float(d[f"xi_css_{r}"]) for r in ratios])
    xf = np.array([float(d[f"xi_csf_{r}"]) for r in ratios])
    a.loglog(ratios, xc, "o-", color=C_ELAS, ms=5, label=r"capillary stress tensor, $\nabla\cdot\Omega$")
    a.loglog(ratios, xf, "s--", color=C_RUPT, ms=5, label=r"CSF, $-\sigma\kappa\nabla c$")
    a.loglog(ratios, xc[0] * (ratios / ratios[0]) ** -2.0, color=C_MUTED, lw=0.8, ls=":", label="second-order slope")
    a.set_xticks(ratios)
    a.set_xticklabels([str(int(r)) for r in ratios])
    a.minorticks_off()
    a.set_xlabel(r"Cells across interface half-width, $\varepsilon/\Delta x$")
    a.set_ylabel(r"Unbalanceable fraction, $\xi=\|\mathbf{F}_{\mathrm{sol}}\|/\|\mathbf{F}\|$")
    a.legend(loc="lower left")
    save(fig, "fig_operator_balance")


def _grid(f):
    lo, hi = f["lo"], f["hi"]
    N = f["eta"].shape[0]
    dx = (hi[0] - lo[0]) / N
    return lo[0] + (np.arange(N) + 0.5) * dx, dx


def _map(ax, fig, data, x, dx, label, **kw):
    ext = np.array([x[0] - dx / 2, x[-1] + dx / 2, x[0] - dx / 2, x[-1] + dx / 2]) * 1e3
    im = ax.imshow(data.T, origin="lower", extent=ext, **kw)
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (mm)")
    ax.set_ylabel("$y$ (mm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label)
    return cb


def fig_fields():
    f = np.load(os.path.join(DATA, "fields_final.npz"))
    x, dx = _grid(f)
    eta = f["eta"]
    # (1) eta + AMR, pressure
    fig, axs = plt.subplots(1, 2, figsize=(TW, 3.1))
    _map(axs[0], fig, eta, x, dx, r"Liquid volume fraction, $\eta$", cmap="Blues", vmin=0, vmax=1)
    cols = {0: "#9a9a9a", 1: C_ELAS, 2: C_RUPT}
    for lev, x0, y0, x1, y1 in f["boxes"]:
        axs[0].add_patch(Rectangle((x0 * 1e3, y0 * 1e3), (x1 - x0) * 1e3, (y1 - y0) * 1e3, fill=False,
                                   ec=cols[int(lev)], lw=0.6 if lev < 2 else 0.9))
    _map(axs[1], fig, f["pressure"] - P_INF, x, dx, r"Gauge pressure, $p-p_\infty$ (Pa)",
         cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-5.0, vmax=30.0))
    axs[1].contour(x * 1e3, x * 1e3, eta.T, levels=[0.5], colors="k", linewidths=0.6)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig_fields_state")
    # (2) Gamma, |u|
    fig, axs = plt.subplots(1, 2, figsize=(TW, 3.1))
    band = eta * (1 - eta) > 1e-3
    G = np.where(band, 1e3 * (f["shell"] - 1.0), np.nan)
    lim = np.nanmax(np.abs(G))
    _map(axs[0], fig, G, x, dx, r"Shell strain, $10^3(\Gamma-1)$", cmap="PuOr_r", vmin=-lim, vmax=lim)
    axs[0].set_xlim(-1.5, 1.5)
    axs[0].set_ylim(-1.5, 1.5)
    for lev, x0, y0, x1, y1 in f["boxes"]:
        if lev == 2:
            axs[0].add_patch(Rectangle((x0 * 1e3, y0 * 1e3), (x1 - x0) * 1e3, (y1 - y0) * 1e3, fill=False,
                                       ec=C_RUPT, lw=0.5, alpha=0.6))
    um = np.hypot(f["velocityx"], f["velocityy"])
    _map(axs[1], fig, um, x, dx, r"Velocity magnitude, $|\mathbf{u}|$ (m s$^{-1}$)", cmap="magma",
         norm=LogNorm(vmin=1e-7, vmax=1e-4))
    axs[1].contour(x * 1e3, x * 1e3, eta.T, levels=[0.5], colors="w", linewidths=0.6)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig_fields_shell")


def fig_profiles():
    R0 = 1.025e-3
    sig0 = CHI * ((R0 / RB) ** 2 - 1)
    figA, a = one(5.6, 3.2)
    figB, b = one(5.6, 3.2)
    for tag, ls in (("early", "--"), ("final", "-")):
        f = np.load(os.path.join(DATA, f"fields_{tag}.npz"))
        x, dx = _grid(f)
        N = len(x)
        j = N // 2
        ray = slice(N // 2, N)
        r = x[ray]
        row = lambda q: 0.5 * (q[ray, j - 1] + q[ray, j])
        t = float(f["t"]) * 1e3
        band = row(f["eta"] * (1 - f["eta"])) > 1e-3
        a.plot((r / R0)[band], 1e3 * (row(f["shell"]) - 1)[band], ls, color=C_ELAS, lw=1.4,
               label=rf"shell strain $10^3(\Gamma-1)$, $t={t:.1f}$ ms")
        a.plot((r / R0)[band], 1e3 * (row(f["kappa2"]) / sig0 - 1)[band], ls, color=C_RUPT, lw=1.4,
               label=rf"tension error $10^3(\sigma_{{\mathrm{{eff}}}}/\sigma_0-1)$, $t={t:.1f}$ ms")
        b.plot(r / R0, row(f["pressure"]) - P_INF, ls, color=C_INK if tag == "final" else C_MUTED, lw=1.4,
               label=rf"simulation, $t={t:.1f}$ ms")
    a.axhline(0, color=C_MUTED, lw=0.6)
    a.set_xlabel(r"Radial position normalised by reference radius, $r/R_0$")
    a.set_ylabel(r"Relative deviation $\times 10^{3}$")
    a.legend(loc="upper right", fontsize=7.8)
    rr = np.linspace(0.0, 2.5, 600) * R0
    b.plot(rr / R0, np.where(rr < R0, sig0 / R0, 0.0), color=C_RUPT, lw=1.0, ls=":", label=r"sharp Laplace jump, $\sigma_0/R_0$")
    b.set_xlim(0, 2.5)
    b.set_xlabel(r"Radial position normalised by reference radius, $r/R_0$")
    b.set_ylabel(r"Gauge pressure, $p-p_\infty$ (Pa)")
    b.legend(loc="center right")
    save(figA, "fig_profile_shell")
    save(figB, "fig_profile_pressure")


def fig_tension(runs):
    col = cmap_R0(runs)
    fig, a = one(6.2, 3.4)
    R = np.linspace(0.82e-3, 1.38e-3, 600)
    a.plot(R * 1e3, sigma_R(R) * 1e3, color=C_INK, lw=1.1, label="Marmottant Law")
    for i, (R0, d) in enumerate(sorted(runs.items())):
        late = d["t"] >= 4.99e-3
        dp = (d["p_in"][late] - d["p_out"][late]).mean()
        a.plot(R0 * 1e3, d["sigma_int"][-1] * 1e3, "o", ms=7, mfc="none", mec=REG_C[regime(R0)], mew=1.3,
               label=r"Interface Average $\langle\sigma_{\mathrm{eff}}\rangle_\eta$" if i == 0 else None)
        a.plot(R0 * 1e3, dp * R0 * 1e3, "x", ms=6, color=C_INK, mew=1.2,
               label=r"Pressure Jump $\overline{\Delta p}\,R_0$" if i == 0 else None)
    a.set_xlabel(r"Initial bubble radius, $R_0$ (mm)")
    a.set_ylabel(r"Surface tension (mN m$^{-1}$)")
    a.legend(loc="upper left")
    save(fig, "fig_tension_map")

    fig, a = one(6.0, 3.6)
    keys = [R0 for R0 in sorted(runs) if regime(R0) != "buckled"]
    for R0 in keys:
        d = runs[R0]
        law0 = min(CHI * ((R0 / RB) ** 2 - 1), SIGW)
        err = np.abs(d["sigma_int"][1:] / law0 - 1.0)
        a.semilogy(d["t"][1:] * 1e3, np.maximum(1e2 * err, 1e-8), LS[regime(R0)], color=col[R0], lw=1.4)
    a.set_ylim(1e-7, 1e-1)
    a.set_xlabel("Time, $t$ (ms)")
    a.set_ylabel(r"Tension error, $100\,|\langle\sigma_{\mathrm{eff}}\rangle_\eta/\sigma(R_0)-1|$  (\%)")
    _regime_legend(a, loc="upper left", regimes=("elastic", "ruptured"))
    _r0_colorbar(fig, a, runs)
    save(fig, "fig_tension_drift")


def _r0_colorbar(fig, ax, runs, label=r"Initial radius $R_0$ (mm)"):
    """Discrete colour bar on the RIGHT mapping each trace to its R_0."""
    keys = sorted(runs)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cb.set_ticks([0.05 + 0.85 * i / (len(keys) - 1) for i in range(len(keys))])
    cb.set_ticklabels([f"{k*1e3:.3f}" for k in keys])
    cb.ax.tick_params(labelsize=7)
    cb.set_label(label, fontsize=9)
    return cb


def _regime_legend(ax, loc="upper left", regimes=("buckled", "elastic", "ruptured")):
    """In-plot key for the line styles actually drawn."""
    for reg in regimes:
        ax.plot([], [], LS[reg], color=C_MUTED, lw=1.5, label=reg.capitalize())
    ax.legend(loc=loc, title="Initial regime", title_fontsize=8.5, fontsize=8.5,
              handlelength=3.0, borderaxespad=0.6)


def fig_radius(runs):
    col = cmap_R0(runs)
    fig, a = one(6.0, 3.6)
    for R0, d in sorted(runs.items()):
        a.plot(d["t"] * 1e3, 1e2 * (d["R_vol"] / d["R_vol"][0] - 1), LS[regime(R0)], color=col[R0], lw=1.3)
    a.set_xlabel("Time, $t$ (ms)")
    a.set_ylabel(r"Radius error, $100\,[R(t)/R(0)-1]$  (%)")
    _regime_legend(a, loc="upper left")
    _r0_colorbar(fig, a, runs)
    save(fig, "fig_radius")

    fig, a = one(6.0, 3.6)
    for R0, d in sorted(runs.items()):
        a.plot(d["t"] * 1e3, d["Gamma_int"] - 1.0, LS[regime(R0)], color=col[R0], lw=1.3)
    a.set_yscale("symlog", linthresh=1e-6)
    a.set_ylim(-1e-1, 1e-3)
    a.set_xlabel("Time, $t$ (ms)")
    a.set_ylabel(r"Shell strain, $\langle\Gamma\rangle_\eta-1$")
    _regime_legend(a, loc="lower left")
    _r0_colorbar(fig, a, runs)
    save(fig, "fig_gamma")


def fig_parasitic(runs, damp):
    col = cmap_R0(runs)
    fig, a = one(5.6, 3.5)
    for R0, d in sorted(runs.items()):
        a.semilogy(d["t"][1:] * 1e3, d["umax"][1:], LS[regime(R0)], color=col[R0], lw=1.3)
    a.set_ylim(5e-6, 5e-2)
    a.set_xlabel("Time, $t$ (ms)")
    a.set_ylabel(r"Maximum speed, $\max|\mathbf{u}|$ (m s$^{-1}$)")
    r0_legend(a, runs, col, loc="upper center", ncol=3)
    save(fig, "fig_parasitic_time")

    fig, a = one(5.6, 3.2)
    k = sorted(runs)
    a.semilogy([r * 1e3 for r in k], [runs[r]["umax"].max() for r in k], "o-", color=C_ELAS, ms=6,
               label=r"$L_{\mathrm{ref}}=\ell/20=0.25$ mm")
    kd = sorted(damp)
    a.semilogy([r * 1e3 for r in kd], [damp[r]["umax"].max() for r in kd], "s--", color=C_RUPT, ms=6,
               label=r"$L_{\mathrm{ref}}=\ell=5$ mm")
    a.axvline(RB * 1e3, color=C_MUTED, lw=0.7, ls=":")
    a.text(RB * 1e3 + 0.005, 3e-5, r"$R_{\mathrm{b}}$", color=C_MUTED)
    a.set_xlabel(r"Initial bubble radius, $R_0$ (mm)")
    a.set_ylabel(r"Peak velocity over 10 ms, $\max_t\max|\mathbf{u}|$ (m s$^{-1}$)")
    a.legend(loc="upper right", title="NSCBC relaxation length", title_fontsize=8.5)
    save(fig, "fig_parasitic_lref")


def fig_lref(ladder):
    fig, a = one(5.6, 3.3)
    cm = plt.get_cmap("viridis")
    for i, (key, Lr) in enumerate([("damp", 5.0e-3), ("L25E3", 2.5e-3), ("L10E3", 1.0e-3), ("L25E4", 2.5e-4)]):
        d = ladder[key]
        a.semilogy(d["t"][1:] * 1e3, d["umax"][1:], color=cm(0.05 + 0.85 * i / 3), lw=1.5,
                   label=rf"$L_{{\mathrm{{ref}}}}={Lr*1e3:g}$ mm ($\ell/{5e-3/Lr:g}$)")
    a.set_xlabel("Time, $t$ (ms)")
    a.set_ylabel(r"Maximum speed, $\max|\mathbf{u}|$ (m s$^{-1}$)")
    a.legend(loc="lower right", title="NSCBC relaxation length", title_fontsize=8.5)
    save(fig, "fig_lref")


def table(runs, damp):
    from scipy.integrate import quad
    fm = lambda v, n=1: f"\\num{{{v:+.{n}e}}}" if v != 0 else "\\num{0}"
    fu = lambda v: f"${v:.1e}$".replace("e-0", "\\,10^{-").replace("e+0", "\\,10^{") + ""
    lines = []
    for R0, d in sorted(runs.items()):
        reg = regime(R0)
        Gb = (R0 / RB) ** 2
        s0 = float(sigma_R(R0))
        late = d["t"] >= 4.99e-3
        dpt = d["p_in"][late] - d["p_out"][late]
        dp, dprms = dpt.mean(), dpt.std()
        J = quad(lambda r: 0.5 / EPS / np.cosh((r - R0) / EPS) ** 2 / r, 1e-9, 20 * R0, points=[R0], limit=500)[0] * R0 - 1
        if s0 > 0:
            S = f"{Gb/(Gb-1):.1f}" if reg == "elastic" else "--"
            e_int = fm(d["sigma_int"][-1] / s0 - 1)
            e_L = f"\\num{{{dp*R0/s0-1:+.4f}}}"
            pred = f"\\num{{{J:+.4f}}}"
        else:
            S = "--"
            e_int = f"(\\num{{{d['sigma_int'][-1]*1e3:.0e}}})" if d['sigma_int'][-1] > 0 else "(0)"
            e_L = f"(${dp:+.0f}\\pm{dprms:.0f}$)"
            pred = "--"
        dR = 100 * (d["R_vol"][-1] / d["R_vol"][0] - 1)
        dG = d["Gamma_int"][-1] - 1
        um = d["umax"].max()
        umd = damp[R0]["umax"].max()
        lines.append(f"{R0*1e3:.3f} & {reg} & {s0*1e3:.2f} & {S} & {e_int} & {e_L} & {pred} & "
                     f"{fm(dG)} & {fm(dR)} & \\num{{{um:.1e}}} & \\num{{{umd:.1e}}} \\\\")
    with open(os.path.join(TAB, "laplace_results.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    runs = load("")
    damp = load("damp_")
    ladder = {}
    for key in ("L25E3", "L10E3", "L25E4"):
        f = os.path.join(DATA, f"laplace_lref_{key}.csv")
        ladder[key] = np.genfromtxt(f, delimiter=",", names=True, skip_header=1)
    ladder["damp"] = damp[8.5e-4]
    fig_law_R(runs)
    fig_law_kin()
    fig_operator()
    fig_fields()
    fig_profiles()
    fig_tension(runs)
    fig_radius(runs)
    fig_parasitic(runs, damp)
    fig_lref(ladder)
    table(runs, damp)
    for key, d in ladder.items():
        print("ladder", key, "peak umax", d["umax"].max(), "final umax", d["umax"][-1], "t_end", d["t"][-1],
              "dR%", 100 * (d["R_vol"][-1] / d["R_vol"][0] - 1))
