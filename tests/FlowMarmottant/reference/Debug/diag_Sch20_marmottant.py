#!/usr/bin/env python3
"""
Damping diagnostics for a coated-bubble (Marmottant) run.

Why this exists: Sch20-Oscillating now tracks the Marmottant-KM reference
through the first collapse, but it over-collapses (R_min ~0.43-0.49 vs 0.52)
and then rings (rebound to ~0.65) where KM is nearly critically damped. The
equilibrium radius is about right, so the elastic balance is fine and the
missing piece is DISSIPATION. That leaves four suspects, and each has its own
measurement here:

  1. Shell dilatational viscosity too weak.
       Compare the measured viscous tension  kappa1 - kappa2  (the solver's
       own pre-floor sigma_tot minus the bare sigma(Gamma)) with the expected
       2 kappa_s Rdot / R.
  2. Elastic tension wrong.
       Compare the measured sigma(Gamma) (kappa2) and Gamma with the
       Marmottant sigma(R) and (R0/R)^2 at the measured radius.
  3. Gas thermodynamics wrong (numerical heating or cooling of the bubble).
       Compare the mean gas pressure with the polytropic p_b (R0/R)^(3 gamma).
  4. Acoustic radiation reflected back instead of leaving.
       KM damps partly by radiating sound. A reflecting boundary returns that
       energy. The pressure profile along the x axis, out to the domain edge,
       is saved for every frame so reflections are visible.

It also writes the terms of the normal-stress balance at the wall, measured
against expected:
    p_gas - p_liq(R+) = 2 sigma/R + 4 mu Rdot/R + 4 kappa_s Rdot/R^2

Usage (run where the plotfiles live, e.g. on INCLINE):
    python3 diag_Sch20_marmottant.py --input ../input_Sch20-Oscillating_Marmottant \\
        --plotdir /mmfs1/.../output_Sch20_Oscillating_Marmottant  [--every 1]

Output, in --out (default ./marm_diag_<run>):
    diag.csv   one row per frame, every scalar below
    diag.npz   the same scalars + x-axis profiles (r, p, u_r, eta) per frame
    diag.png   a 6-panel summary
Everything is small (a few MB), so copy the folder back and send it.

Memory: each frame loads only a covering grid of the bubble region at the
finest level, capped at --max-cells per side (default 160), plus one 1-D ray.
"""
import argparse, csv, glob, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import marmottant_sch20_common as M                                     # noqa: E402


# --------------------------------------------------------------------------- #
def plotfiles(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*cell")) if os.path.isdir(p))


def field_or_none(obj, name):
    try:
        return np.asarray(obj[name], dtype=float)
    except Exception:
        return None


def pick_level(ds, window, max_cells):
    """Finest level whose covering grid over `window` stays under max_cells/side."""
    L = ds.domain_width[0].v / ds.domain_dimensions[0]
    for lev in range(ds.index.max_level, -1, -1):
        dx = L / 2 ** lev
        if window / dx <= max_cells:
            return lev, dx
    return 0, L


def frame_stats(pf, R0, kappa_s, mu_l, sym, win_fac, max_cells):
    import yt
    yt.funcs.mylog.setLevel(50)
    ds = yt.load(pf)
    t = float(ds.current_time)
    lo = np.array([float(v) for v in ds.domain_left_edge])
    hi = np.array([float(v) for v in ds.domain_right_edge])
    window = win_fac * R0
    lev, dx = pick_level(ds, window, max_cells)
    whi = np.minimum(hi, lo + window)
    dims = np.maximum(1, np.round((whi - lo) / dx).astype(int))
    cg = ds.covering_grid(level=lev, left_edge=lo, dims=dims)

    eta = field_or_none(cg, "eta")
    p = field_or_none(cg, "pressure")
    ux = field_or_none(cg, "velocityx")
    uy = field_or_none(cg, "velocityy")
    uz = field_or_none(cg, "velocityz")
    G = field_or_none(cg, "Gamma")
    k1 = field_or_none(cg, "kappa1")
    k2 = field_or_none(cg, "kappa2")
    rho = field_or_none(cg, "density")
    gx, gy, gz = (field_or_none(cg, "grad_eta" + c) for c in "xyz")

    # cell-centre coordinates
    ax = [lo[d] + (np.arange(dims[d]) + 0.5) * dx for d in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z) + 1e-300

    out = dict(t=t, level=lev, dx=dx)

    # ---- radii ----------------------------------------------------------- #
    vol = float(np.sum(1.0 - eta)) * dx ** 3 * sym          # eta = 1 is liquid
    RV = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    out["R_V"] = RV
    nb = max(40, int(window / dx))
    edges = np.linspace(0.0, window, nb + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    eb = np.bincount(idx, weights=eta.ravel(), minlength=nb) / np.maximum(cnt, 1)
    rc = 0.5 * (edges[1:] + edges[:-1])
    ok = cnt > 0
    R05 = np.nan
    rr, ee = rc[ok], eb[ok]
    for i in range(1, len(rr)):
        if ee[i - 1] < 0.5 <= ee[i]:
            R05 = rr[i - 1] + (0.5 - ee[i - 1]) * (rr[i] - rr[i - 1]) / (ee[i] - ee[i - 1])
            break
    out["R_05"] = R05

    # ---- interface band weights ----------------------------------------- #
    if gx is not None:
        w = np.sqrt(gx * gx + gy * gy + gz * gz)
    else:
        w = np.sqrt(sum(g * g for g in np.gradient(eta, dx)))
    band = w > 0.05 * w.max() if w.max() > 0 else np.zeros_like(w, dtype=bool)
    wb = w[band]
    # frame 0 is written before grad_eta is first filled, so the band can be empty
    bavg = lambda q: (float(np.sum(q[band] * wb) / np.sum(wb))
                      if (q is not None and wb.size and np.sum(wb) > 0) else np.nan)

    ur = None
    if ux is not None:
        ur = (X * ux + Y * uy + Z * uz) / r
    out["Rdot_band"] = bavg(ur)          # |grad eta|-weighted u_r at the wall
    out["Gamma"] = bavg(G)
    out["sig_elastic"] = bavg(k2)        # sigma(Gamma), measured
    out["sig_total"] = bavg(k1)          # sigma + kappa_s div_s u, pre-floor
    out["sig_visc"] = (out["sig_total"] - out["sig_elastic"]) if k1 is not None else np.nan
    out["band_cells"] = int(band.sum())
    out["band_width_cells"] = float(np.sum(eta[(eta > 0.05) & (eta < 0.95)] >= 0) /
                                     max(4.0 * np.pi * RV ** 2 / sym / dx ** 2, 1.0))

    # ---- pressures ------------------------------------------------------ #
    gas = eta < 0.05
    out["p_gas_mean"] = float(np.mean(p[gas])) if gas.any() else np.nan
    out["p_gas_centre"] = float(p[0, 0, 0])
    # liquid just outside the band: shell R_05 + [3, 6] dx
    Rref = R05 if np.isfinite(R05) else RV
    shell = (r > Rref + 3 * dx) & (r < Rref + 6 * dx) & (eta > 0.95)
    out["p_liq_wall"] = float(np.mean(p[shell])) if shell.any() else np.nan
    out["dp_wall_meas"] = out["p_gas_mean"] - out["p_liq_wall"]

    # ---- kinetic energy in the window (octant x sym) --------------------- #
    if rho is not None and ux is not None:
        ke = 0.5 * rho * (ux * ux + uy * uy + uz * uz)
        out["KE_window"] = float(np.sum(ke)) * dx ** 3 * sym
        out["KE_gas"] = float(np.sum(ke[gas])) * dx ** 3 * sym
    else:
        out["KE_window"] = out["KE_gas"] = np.nan

    # ---- 1-D profile along x, whole domain (acoustic reflections) -------- #
    prof = None
    try:
        off = 0.25 * dx
        ray = ds.ortho_ray(0, (lo[1] + off, lo[2] + off))
        xs = np.asarray(ray["index", "x"], float)
        order = np.argsort(xs)
        prof = dict(x=xs[order],
                    p=np.asarray(ray["pressure"], float)[order],
                    ux=np.asarray(ray["velocityx"], float)[order],
                    eta=np.asarray(ray["eta"], float)[order])
        out["p_far_edge"] = float(prof["p"][-1])
        mid = np.argmin(np.abs(prof["x"] - 0.5 * (lo[0] + hi[0])))
        out["p_far_mid"] = float(prof["p"][mid])
    except Exception as exc:                                    # noqa: BLE001
        out["p_far_edge"] = out["p_far_mid"] = np.nan
        print("    [ray failed: %s]" % exc)
    return out, prof


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="the run's input file")
    ap.add_argument("--plotdir", default=None,
                    help="directory holding NNNNNcell (default: plot_file from the input)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--window", type=float, default=1.4,
                    help="covering-grid half-width in units of R0 (default 1.4)")
    ap.add_argument("--max-cells", type=int, default=160,
                    help="cap on covering-grid cells per side (memory guard)")
    a = ap.parse_args()

    kv = M.parse_input(a.input)
    shell, liq, gas, meta = M.build_case(kv, a.input)
    R0 = meta["R0"]
    plotdir = a.plotdir or meta["plot_file"]
    frames = plotfiles(plotdir)[::max(a.every, 1)]
    if not frames:
        sys.exit("no NNNNNcell plotfiles in %s" % plotdir)
    sym = 2 ** sum(1 for v in meta["prob_lo"] if abs(v) < 1e-30)  # octant -> 8
    out_dir = a.out or os.path.join(os.getcwd(), "marm_diag_" + meta["name"])
    os.makedirs(out_dir, exist_ok=True)
    print("input   : %s\nplotdir : %s\nframes  : %d   octant factor %d\nout     : %s"
          % (a.input, plotdir, len(frames), sym, out_dir))
    print("R0=%.3e  kappa_s=%.3e  chi=%.3g  mu=%.3e  p_inf=%.3e  p_b=%.3e  tau_c=%.3e"
          % (R0, shell.kappa_s, shell.chi, liq.mu, liq.p_inf, gas.p_g0, meta["tau_c"]))

    rows, profs = [], []
    for k, pf in enumerate(frames):
        try:
            s, prof = frame_stats(pf, R0, shell.kappa_s, liq.mu, sym, a.window, a.max_cells)
        except Exception as exc:                                # noqa: BLE001
            print("  %s: FAILED %s" % (os.path.basename(pf), exc))
            continue
        s["frame"] = os.path.basename(pf)
        rows.append(s)
        profs.append(prof)
        print("  %s t/tc=%.3f  R_V/R0=%.4f  R_05/R0=%.4f  Rdot=%+.3f  sig_el=%.4f  sig_v=%+.4f"
              % (s["frame"], s["t"] / meta["tau_c"], s["R_V"] / R0, s["R_05"] / R0,
                 s["Rdot_band"], s["sig_elastic"], s["sig_visc"]))

    if not rows:
        sys.exit("no frames could be read")

    # ---- expected terms at the MEASURED radius --------------------------- #
    t = np.array([r["t"] for r in rows])
    RV = np.array([r["R_V"] for r in rows])
    Rd_fd = np.gradient(RV, t) if len(t) > 2 else np.full_like(RV, np.nan)
    for r_, rdfd in zip(rows, Rd_fd):
        R = r_["R_V"]
        Rd = r_["Rdot_band"]
        r_["Rdot_fd"] = float(rdfd)
        r_["Gamma_expected"] = (R0 / R) ** 2
        r_["sig_elastic_expected"] = float(shell.sigma(R))
        r_["sig_visc_expected_band"] = 2.0 * shell.kappa_s * Rd / R
        r_["sig_visc_expected_fd"] = 2.0 * shell.kappa_s * rdfd / R
        r_["p_gas_polytropic"] = gas.p_g0 * (R0 / R) ** (3.0 * gas.kappa_g)
        r_["dp_wall_expected"] = (2.0 * shell.sigma(R) / R + 4.0 * liq.mu * Rd / R
                                  + 4.0 * shell.kappa_s * Rd / R ** 2)
        r_["term_laplace"] = 2.0 * shell.sigma(R) / R
        r_["term_mu"] = 4.0 * liq.mu * Rd / R
        r_["term_kappa_s"] = 4.0 * shell.kappa_s * Rd / R ** 2
        r_["term_kappa_s_meas"] = 2.0 * r_["sig_visc"] / R
        r_["KE_rayleigh"] = 2.0 * np.pi * liq.rho * R ** 3 * Rd ** 2

    keys = list(rows[0].keys())
    with open(os.path.join(out_dir, "diag.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r_ in rows:
            wr.writerow(r_)

    npz = {k: np.array([r_.get(k, np.nan) for r_ in rows]) for k in keys if k != "frame"}
    npz["frame"] = np.array([r_["frame"] for r_ in rows])
    for j, pr in enumerate(profs):
        if pr is None:
            continue
        for k, v in pr.items():
            npz["prof%03d_%s" % (j, k)] = v
    npz["meta_R0"] = R0
    npz["meta_tau_c"] = meta["tau_c"]
    npz["meta_kappa_s"] = shell.kappa_s
    npz["meta_chi"] = shell.chi
    npz["meta_R_buck"] = shell.R_buck
    npz["meta_mu"] = liq.mu
    npz["meta_rho"] = liq.rho
    npz["meta_c"] = liq.c
    npz["meta_p_inf"] = liq.p_inf
    npz["meta_p_b"] = gas.p_g0
    npz["meta_gamma_g"] = gas.kappa_g
    npz["meta_max_level"] = meta["max_level"]
    npz["meta_epsilon"] = meta["epsilon"] if meta["epsilon"] else np.nan
    np.savez_compressed(os.path.join(out_dir, "diag.npz"), **npz)

    # ---- KM reference on the same clock ---------------------------------- #
    try:
        t_km, R_km, Rd_km = M.solve_km(R0, 0.0, (0.0, float(t[-1])), shell, liq, gas,
                                       t_eval=np.linspace(0.0, float(t[-1]), 2000))
    except Exception:
        t_km = R_km = Rd_km = None

    summarize(rows, meta, shell, liq, gas, t_km, R_km, Rd_km)
    make_plot(rows, meta, out_dir, t_km, R_km, Rd_km)
    print("\nwrote %s/{diag.csv, diag.npz, diag.png}" % out_dir)


def summarize(rows, meta, shell, liq, gas, t_km, R_km, Rd_km):
    R0, tc = meta["R0"], meta["tau_c"]
    g = lambda k: np.array([r[k] for r in rows], float)
    RV, t = g("R_V"), g("t")
    i = int(np.argmin(RV))
    print("\n================ SUMMARY ================")
    print("R_min/R0 : R_V %.4f (t/tc %.3f)   R_05 %.4f" % (RV[i] / R0, t[i] / tc, g("R_05")[i] / R0))
    if R_km is not None:
        j = int(np.argmin(R_km))
        print("R_min/R0 : KM  %.4f (t/tc %.3f)" % (R_km[j] / R0, t_km[j] / tc))

    def ratio(a, b):
        m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12 * np.nanmax(np.abs(b)))
        return np.median(a[m] / b[m]) if m.any() else np.nan

    moving = np.abs(g("Rdot_band")) > 0.1 * np.nanmax(np.abs(g("Rdot_band")))
    sv, sve = g("sig_visc")[moving], g("sig_visc_expected_band")[moving]
    print("\n(1) shell viscous tension, measured / expected 2 kappa_s Rdot/R")
    print("      median ratio  %.3f   (1.0 = correct; <1 = too little damping)" % ratio(sv, sve))
    print("      same vs finite-difference Rdot: %.3f"
          % ratio(sv, g("sig_visc_expected_fd")[moving]))
    print("      Rdot at band / dR_V/dt  median %.3f" % ratio(g("Rdot_band"), g("Rdot_fd")))
    print("(2) elastic: Gamma / (R0/R)^2  median %.4f ;  sigma(Gamma)-sigma(R) max |diff| %.4f N/m"
          % (ratio(g("Gamma"), g("Gamma_expected")),
             np.nanmax(np.abs(g("sig_elastic") - g("sig_elastic_expected")))))
    print("(3) gas: p_gas_mean / polytropic  at R_min %.3f,  median %.3f"
          % (g("p_gas_mean")[i] / g("p_gas_polytropic")[i], ratio(g("p_gas_mean"), g("p_gas_polytropic"))))
    print("(4) far field: p at x-edge  min %.4g  max %.4g  (p_inf %.4g)"
          % (np.nanmin(g("p_far_edge")), np.nanmax(g("p_far_edge")), liq.p_inf))
    print("    wall jump: measured / expected  median %.3f" % ratio(g("dp_wall_meas"), g("dp_wall_expected")))
    print("    KE window / Rayleigh 2 pi rho R^3 Rdot^2  median %.3f" % ratio(g("KE_window"), g("KE_rayleigh")))


def make_plot(rows, meta, out_dir, t_km, R_km, Rd_km):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    R0, tc = meta["R0"], meta["tau_c"]
    g = lambda k: np.array([r[k] for r in rows], float)
    t = g("t") / tc
    fig, axs = plt.subplots(3, 2, figsize=(12, 11))
    a = axs[0, 0]
    a.plot(t, g("R_V") / R0, "o-", ms=3, label="R_V")
    a.plot(t, g("R_05") / R0, "s-", ms=3, label="R_0.5")
    if R_km is not None:
        a.plot(t_km / tc, R_km / R0, "k-", label="KM full shell")
    a.set_ylabel("R/R0"); a.legend(); a.set_title("radius")
    a = axs[0, 1]
    a.plot(t, g("Rdot_band"), "o-", ms=3, label="u_r at band")
    a.plot(t, g("Rdot_fd"), "s-", ms=3, label="dR_V/dt")
    if Rd_km is not None:
        a.plot(t_km / tc, Rd_km, "k-", label="KM")
    a.set_ylabel("Rdot [m/s]"); a.legend(); a.set_title("wall velocity")
    a = axs[1, 0]
    a.plot(t, g("sig_visc"), "o-", ms=3, label="measured kappa1-kappa2")
    a.plot(t, g("sig_visc_expected_band"), "k--", label="2 kappa_s Rdot/R")
    a.set_ylabel("N/m"); a.legend(); a.set_title("(1) shell viscous tension")
    a = axs[1, 1]
    a.plot(t, g("sig_elastic"), "o-", ms=3, label="sigma(Gamma) measured")
    a.plot(t, g("sig_elastic_expected"), "k--", label="sigma(R) Marmottant")
    a2 = a.twinx()
    a2.plot(t, g("Gamma") / g("Gamma_expected"), "r:", label="Gamma/(R0/R)^2")
    a2.set_ylabel("Gamma ratio", color="r")
    a.set_ylabel("N/m"); a.legend(loc="upper left"); a.set_title("(2) elastic tension")
    a = axs[2, 0]
    a.semilogy(t, g("p_gas_mean"), "o-", ms=3, label="p_gas measured")
    a.semilogy(t, g("p_gas_polytropic"), "k--", label="p_b (R0/R)^(3 gamma)")
    a.set_ylabel("Pa"); a.set_xlabel("t/tc"); a.legend(); a.set_title("(3) gas pressure")
    a = axs[2, 1]
    a.plot(t, g("p_far_edge") / meta_p_inf(rows, meta), "o-", ms=3, label="p(x edge)/p_inf")
    a.plot(t, g("p_far_mid") / meta_p_inf(rows, meta), "s-", ms=3, label="p(x mid)/p_inf")
    a.set_xlabel("t/tc"); a.legend(); a.set_title("(4) far-field pressure")
    for a in axs.flat:
        a.grid(alpha=0.3)
    fig.suptitle(meta["name"] + ": damping diagnostics")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diag.png"), dpi=130)


def meta_p_inf(rows, meta):
    kv = M.parse_input(meta["input_path"])
    _, liq, _, _ = M.build_case(kv, meta["input_path"])
    return liq.p_inf


if __name__ == "__main__":
    main()
