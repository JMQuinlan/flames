#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
DRIVEN BUBBLE -- DEBUG / DIAGNOSTIC ANALYSIS
===============================================================================
Built to answer ONE question: why does the simulated driven bubble sit ~0.13%
below the driven RPE/KM reference with ~12% less oscillation amplitude, after
the finite-domain (confinement) correction has already removed the phase error?

It computes every quantity needed to separate the competing explanations, and
writes a COMPACT summary (.npz + .txt) so the result can be moved off the
cluster without shipping plotfiles.

WHAT IT MEASURES, AND WHICH HYPOTHESIS EACH ONE KILLS
  1. CONSERVATION.  Gas mass, liquid mass, total mass and total energy vs time.
     apply_vaporization = 0 in these runs, so any gas-mass loss is numerical
     (volume-fraction diffusion, or the stiff relaxation moving mass across the
     band).  A -0.13% radius offset is -0.4% in volume: if gas mass falls by
     that much, the offset is explained outright.
  2. RADIUS MEASURES.  R_V (gas volume) and R_0.5 (radially averaged eta=0.5
     contour) together.  R_V/R0 already starts at ~1.0034 because the diffuse
     band biases a volume integral.  If the two DIVERGE the offset is a
     measurement artifact (band drift); if they move TOGETHER the bubble is
     really shrinking.
  3. BAND WIDTH.  Interface thickness from the eta profile.  Distinguishes
     "band is diffusing" from "bubble is shrinking" directly.
  4. RESIDUAL DECOMPOSITION, per drive cycle: mean offset, amplitude ratio and
     phase shift against the reference.  Shows whether the error is a constant
     bias or accumulates cycle by cycle.
  5. PRESSURE PROBES vs RADIUS.  Pressure amplitude at r/R0 = 0, 1.5, 2, 3 and
     near the wall.  The existing drive probe reads the ORIGIN, which is INSIDE
     the bubble, so it cannot distinguish imposed forcing from bubble response
     -- which is why the "7.36x box amplification" number is unreliable.  A
     probe in pure liquid away from the interface separates them.  Note the
     contradiction this has to resolve: a gain > 1 means the gas is driven
     HARDER than the model assumes, so the sim should respond MORE, yet it
     responds 12% LESS.
  6. ENERGY BUDGET.  Kinetic vs internal, to see whether the missing
     oscillation energy is dissipated (numerical damping) or never delivered.

USAGE
    python3 driven_debug.py --input ../input_f81.5_A1.36
    python3 driven_debug.py --input ../input_f25.0_A1.00 --output /path/to/plotfiles
    python3 driven_debug.py --input ... --every 2      # subsample frames

OUTPUT
    Images/driven_debug_<case>.png / .eps     four-panel diagnostic
    driven_debug_<case>.npz                   all series (small -- transfer this)
    driven_debug_<case>.txt                   the printed report
===============================================================================
"""
import argparse, glob, math, os, re, sys

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message=".*PostScript backend.*")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(_HERE, "Images")
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))

FIG_W, FIG_H, DPI = 13, 9.0, 170
_LOG = []


def say(s=""):
    print(s)
    _LOG.append(s)


# ---------------------------------------------------------------- input ----
def parse_input(path):
    kv = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def num(kv, key, default=None):
    if key not in kv:
        return default
    raw = kv[key]
    seen = 0
    while raw in kv and seen < 8:
        raw = kv[raw]; seen += 1
    try:
        return float(raw.split()[0].strip('"'))
    except ValueError:
        return default


def drive_of(kv):
    """(A, omega, phase) from the first driven nscbc face; arrays -> first term."""
    for face in ("xhi", "yhi", "zhi", "xlo", "ylo", "zlo"):
        a = kv.get(f"nscbc.{face}.drive_amp"); w = kv.get(f"nscbc.{face}.drive_omega")
        if a and w:
            A = [float(t) for t in a.split()]
            W = [float(t) for t in w.split()]
            P = [float(t) for t in kv.get(f"nscbc.{face}.drive_phase", "").split()] or [0.0]*len(A)
            if any(x != 0 for x in A):
                return list(zip(A, W, P))
    return []


def resolve_out(kv, override):
    cands = [override, kv.get("plot_file", "")]
    base = os.path.basename((override or kv.get("plot_file", "")).rstrip("/"))
    if base:
        cands += [os.path.join(_REPO, "bin", "tests", "FlowDrivenBubble", base),
                  os.path.join(_REPO, "bin", "tests", "FlowDrivenBubble", "domaintest", base)]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return None


# ------------------------------------------------------------ extraction ----
def extract(pfdir, R0, every=1, nbins=400):
    """One pass per plotfile; returns a dict of series."""
    import yt
    yt.funcs.mylog.setLevel(50)
    pfs = sorted(glob.glob(os.path.join(pfdir, "*cell")),
                 key=lambda s: int(re.search(r"(\d+)cell", s).group(1)))[::every]
    S = {k: [] for k in ("t", "R_V", "R_05", "m_gas", "m_liq", "m_tot",
                         "E_tot", "KE", "UE", "band", "p_probe", "src", "R_m", "rho_c", "sym")}
    for pf in pfs:
        try:
            ds = yt.load(pf); ad = ds.all_data()
            dle, dre = ds.domain_left_edge, ds.domain_right_edge
            sym, cen = 1, []
            for d in range(3):
                if float(dle[d]) > -1e-6:
                    sym *= 2; cen.append(0.0)
                else:
                    cen.append(0.5*float(dle[d] + dre[d]))
            eta = np.array(ad["eta"], float)
            try:   vol = np.array(ad["index", "cell_volume"], float)
            except Exception: vol = np.array(ad["cell_volume"], float)
            g = np.clip(1.0 - eta, 0.0, 1.0)

            # ---- masses (rho_eta1 = gas here: eta=0 inside the bubble) ----
            def fld(n, default=None):
                try:    return np.array(ad[n], float)
                except Exception: return default
            r1 = fld("rho_eta1"); r0 = fld("rho_eta0")
            m_gas = float(np.sum(r1*vol))*sym if r1 is not None else np.nan
            m_liq = float(np.sum(r0*vol))*sym if r0 is not None else np.nan
            rho = fld("density")
            m_tot = float(np.sum(rho*vol))*sym if rho is not None else np.nan

            # ---- energies ----
            # KE_per_vol / UE_per_vol are not written by this integrator, so build
            # KE from density and velocity and take UE as the remainder.
            Ev = fld("energy_per_vol")
            E_tot = float(np.sum(Ev*vol))*sym if Ev is not None else np.nan
            KEv = fld("KE_per_vol")
            if KEv is None and rho is not None:
                u = fld("velocityx"); v = fld("velocityy"); w_ = fld("velocityz")
                if u is not None:
                    q = u*u
                    if v is not None: q = q + v*v
                    if w_ is not None: q = q + w_*w_
                    KEv = 0.5*rho*q
            KE = float(np.sum(KEv*vol))*sym if KEv is not None else np.nan
            UEv = fld("UE_per_vol")
            if UEv is None and Ev is not None and KEv is not None:
                UEv = Ev - KEv
            UE = float(np.sum(UEv*vol))*sym if UEv is not None else np.nan

            # ---- explicit mass source (should be ~0: apply_vaporization = 0) ----
            sr = fld("Source_rho")
            src = float(np.sum(sr*vol))*sym if sr is not None else np.nan

            # ---- radii ----
            V_b = float(np.sum(g*vol))*sym
            R_V = (3.0*V_b/(4.0*math.pi))**(1.0/3.0)
            x = np.array(ad["x"], float) - cen[0]
            y = np.array(ad["y"], float) - cen[1]
            z = np.array(ad["z"], float) - cen[2]
            r = np.sqrt(x*x + y*y + z*z)
            rmax = min(float(r.max()), 10.0*R0)
            edges = np.linspace(0.0, rmax, nbins+1)
            idx = np.clip(np.digitize(r, edges)-1, 0, nbins-1)
            rc = 0.5*(edges[:-1] + edges[1:])
            w = np.bincount(idx, weights=vol, minlength=nbins)
            e = np.bincount(idx, weights=eta*vol, minlength=nbins)
            ok = w > 0
            rr, ee = rc[ok], e[ok]/w[ok]
            R_05 = _cross(rr, ee, 0.5)
            # Mass-moment radius: gas mass is conserved EXACTLY, and a smearing
            # band moves mass symmetrically about the interface, so the second
            # moment of the gas mass is far less biased than int(1-eta)dV.
            # For a uniform sphere <r^2> = (3/5) R^2.
            R_m = np.nan
            if r1 is not None:
                mw = r1*vol; msum = float(np.sum(mw))
                if msum > 0:
                    R_m = math.sqrt(5.0/3.0*float(np.sum(mw*r*r))/msum)
            # core gas density (eta < 0.1 -> essentially pure gas)
            core = eta < 0.1
            rho_c = (float(np.sum((r1[core]/np.maximum(1.0-eta[core], 1e-12))*vol[core])
                           / np.sum(vol[core]))
                     if (r1 is not None and core.any()) else np.nan)
            # band width: radial distance between eta=0.1 and eta=0.9
            band = _cross(rr, ee, 0.9) - _cross(rr, ee, 0.1)

            # ---- pressure probes in shells (pure liquid away from the band) ----
            p = fld("pressure")
            probes = []
            for rp in (0.0, 1.5, 2.0, 3.0):
                if rp == 0.0:
                    m = r < 0.3*R0
                else:
                    m = (r > (rp-0.15)*R0) & (r < (rp+0.15)*R0)
                probes.append(float(p[m].mean()) if (p is not None and m.any()) else np.nan)
            m = r > 0.85*rmax
            probes.append(float(p[m].mean()) if (p is not None and m.any()) else np.nan)

            S["t"].append(float(ds.current_time)); S["R_V"].append(R_V)
            S["R_05"].append(R_05); S["m_gas"].append(m_gas); S["m_liq"].append(m_liq)
            S["m_tot"].append(m_tot); S["E_tot"].append(E_tot); S["KE"].append(KE)
            S["UE"].append(UE); S["band"].append(band); S["p_probe"].append(probes)
            S["src"].append(src); S["R_m"].append(R_m); S["rho_c"].append(rho_c)
            S["sym"].append(sym)
        except Exception as exc:
            say(f"  [warn] skipped {os.path.basename(pf)}: {exc}")
    for k in S:
        S[k] = np.array(S[k], dtype=float)
    if len(S["t"]):
        o = np.argsort(S["t"])
        for k in S:
            S[k] = S[k][o]
    return S


def _cross(rr, ee, lev):
    for i in range(len(rr)-1):
        if (ee[i]-lev)*(ee[i+1]-lev) < 0:
            return rr[i] + (lev-ee[i])*(rr[i+1]-rr[i])/(ee[i+1]-ee[i])
    return np.nan


# ------------------------------------------------------------- reporting ----
def drift_table(S, R0):
    say("\n" + "="*78)
    say("1. CONSERVATION  (apply_vaporization = 0: any gas-mass loss is NUMERICAL)")
    say("="*78)
    if not len(S["t"]) or not np.isfinite(S["m_gas"]).any():
        say("   no mass fields available"); return
    g0 = S["m_gas"][0]; t0 = S["m_tot"][0]; e0 = S["E_tot"][0]
    say(f"{'t':>11} {'m_gas':>13} {'d m_gas %':>11} {'d m_tot %':>11} {'d E_tot %':>11}")
    for i in range(0, len(S["t"]), max(1, len(S["t"])//12)):
        say(f"{S['t'][i]:11.4e} {S['m_gas'][i]:13.6e} "
            f"{100*(S['m_gas'][i]/g0-1):11.5f} "
            f"{100*(S['m_tot'][i]/t0-1) if np.isfinite(t0) else float('nan'):11.5f} "
            f"{100*(S['E_tot'][i]/e0-1) if np.isfinite(e0) else float('nan'):11.5f}")
    if np.isfinite(S["src"]).any():
        say(f"\n   volumetric mass source int(Source_rho)dV : "
            f"max |.| = {np.nanmax(np.abs(S['src'])):.4e}")
        say("   >> apply_vaporization = 0, so this should be ~0.  If it is not,")
        say("      the mass loss is an explicit source, not numerical diffusion.")
    dg = 100*(S["m_gas"][-1]/g0 - 1)
    say(f"\n   gas-mass drift over the run : {dg:+.5f} %")
    say(f"   implied radius offset       : {dg/3.0:+.5f} %   (dR/R = dV/3V)")
    say("   >> compare with the measured mean offset (~ -0.13%).  If they match,")
    say("      the offset is gas leaking across the band, not a physics error.")


def radius_table(S, R0):
    say("\n" + "="*78)
    say("2. RADIUS MEASURES + 3. BAND WIDTH")
    say("="*78)
    if not len(S["t"]): return
    say(f"{'t':>11} {'R_V/R0':>10} {'R_0.5/R0':>10} {'R_m/R0':>10} "
        f"{'(R_V-R_0.5)/R0 %':>17} {'band/R0':>10}")
    for i in range(0, len(S["t"]), max(1, len(S["t"])//12)):
        say(f"{S['t'][i]:11.4e} {S['R_V'][i]/R0:10.6f} {S['R_05'][i]/R0:10.6f} "
            f"{S['R_m'][i]/R0:10.6f} "
            f"{100*(S['R_V'][i]-S['R_05'][i])/R0:17.5f} {S['band'][i]/R0:10.5f}")
    d = (S["R_V"]-S["R_05"])/R0
    b = S["band"]/R0
    say(f"\n   (R_V - R_0.5)/R0 : start {100*d[0]:+.5f} %   end {100*d[-1]:+.5f} %   "
        f"change {100*(d[-1]-d[0]):+.5f} %")
    if np.isfinite(b).any():
        say(f"   band width / R0  : start {b[0]:.5f}      end {b[-1]:.5f}      "
            f"change {100*(b[-1]/b[0]-1):+.3f} %")
    if np.isfinite(S["R_m"]).any():
        say(f"   R_m/R0 (mass moment): start {S['R_m'][0]/R0:.6f}   "
            f"end {S['R_m'][-1]/R0:.6f}   drift {100*(S['R_m'][-1]/S['R_m'][0]-1):+.5f} %")
        say("   >> R_m is built from the EXACTLY conserved gas mass.  If R_m is")
        say("      flat while R_V and R_0.5 fall, the drift is an eta-field")
        say("      smearing artifact, not a shrinking bubble.")
    if np.isfinite(S["rho_c"]).any():
        say(f"   core gas density   : start {S['rho_c'][0]:.6f}   "
            f"end {S['rho_c'][-1]:.6f}   drift {100*(S['rho_c'][-1]/S['rho_c'][0]-1):+.5f} %")
        say("   >> gas mass is constant, so REAL shrinkage must raise this.")
    say("   >> DIVERGING  -> the offset is a band/measurement artifact.")
    say("   >> TOGETHER   -> the bubble is genuinely shrinking (see table 1).")


def probe_table(S, R0, terms, p_inf):
    say("\n" + "="*78)
    say("5. PRESSURE PROBES vs RADIUS  (separates FORCING from RESPONSE)")
    say("="*78)
    if not len(S["t"]) or S["p_probe"].ndim != 2:
        say("   no pressure field available"); return
    A = sum(abs(a) for a, _, _ in terms) if terms else float("nan")
    lbl = ["origin (r<0.3R0, INSIDE the bubble)", "r = 1.5 R0", "r = 2 R0",
           "r = 3 R0", "near wall"]
    say(f"   wall drive amplitude A = {A:.4g} Pa")
    say(f"\n{'probe':38} {'mean p':>13} {'osc amp':>12} {'amp/A':>9}")
    for j, name in enumerate(lbl):
        col = S["p_probe"][:, j]
        if not np.isfinite(col).any():
            continue
        amp = 0.5*(np.nanmax(col) - np.nanmin(col))
        say(f"{name:38} {np.nanmean(col):13.5f} {amp:12.5e} {amp/A:9.4f}")
    say("\n   >> the ORIGIN probe sits inside the bubble: its ratio mixes the")
    say("      imposed forcing with the bubble's own response.  The r >= 2 R0")
    say("      probes are in pure liquid and are the ones to trust for the")
    say("      'box amplification' factor.")
    say("   >> if amp/A at r = 2-3 R0 is ~1, there is NO amplification and the")
    say("      7.36 calibration in the inputs is wrong.")


def residual_table(S, R0, terms, P):
    """Per-cycle offset / amplitude / phase against the driven reference."""
    say("\n" + "="*78)
    say("4. RESIDUAL DECOMPOSITION vs the driven reference, PER CYCLE")
    say("="*78)
    if P is None or not len(S["t"]):
        say("   reference model unavailable (analyze_radius_drivencompare not importable)")
        return
    tm, Rm = P
    t, R = S["t"], S["R_V"]
    Ri = np.interp(t, tm, Rm)
    s = R/R[0] - 1.0
    m = Ri/Ri[0] - 1.0
    T = 2*math.pi/terms[0][1]
    say(f"   drive period T = {T:.5e} s;  {t[-1]/T:.2f} cycles of data")
    say(f"\n{'cycle':>7} {'offset %':>11} {'amp ratio':>11} {'rms %':>9}")
    nc = max(1, int(t[-1]/T))
    for c in range(nc):
        k = (t >= c*T) & (t < (c+1)*T)
        if k.sum() < 4:
            continue
        ss, mm = s[k], m[k]
        off = ss.mean() - mm.mean()
        den = np.dot(mm-mm.mean(), mm-mm.mean())
        amp = np.dot(ss-ss.mean(), mm-mm.mean())/den if den > 0 else np.nan
        say(f"{c:7d} {100*off:11.5f} {amp:11.4f} {100*np.sqrt(np.mean((ss-mm)**2)):9.4f}")
    say("\n   >> offset GROWING with cycle  -> accumulating drift (mass loss).")
    say("   >> offset CONSTANT            -> a fixed bias (calibration/geometry).")
    say("   >> amp ratio < 1 and constant -> numerical dissipation.")


def energy_table(S):
    say("\n" + "="*78)
    say("6. ENERGY BUDGET")
    say("="*78)
    if not len(S["t"]) or not np.isfinite(S["KE"]).any():
        say("   no energy fields available"); return
    say(f"{'t':>11} {'KE':>13} {'UE':>13} {'E_tot':>13} {'d E_tot %':>11}")
    e0 = S["E_tot"][0]
    for i in range(0, len(S["t"]), max(1, len(S["t"])//12)):
        say(f"{S['t'][i]:11.4e} {S['KE'][i]:13.5e} {S['UE'][i]:13.5e} "
            f"{S['E_tot'][i]:13.6e} {100*(S['E_tot'][i]/e0-1):11.5f}")
    say("\n   >> KE peaks should decay if the oscillation is being damped.")


def plots(S, R0, stem, P=None):
    if not len(S["t"]):
        return
    os.makedirs(IMG, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(FIG_W, FIG_H))
    t = S["t"]
    a = ax[0][0]
    if np.isfinite(S["m_gas"]).any():
        a.plot(t, 100*(S["m_gas"]/S["m_gas"][0]-1), "-o", ms=3, color="tab:red",
               label="gas mass")
    if np.isfinite(S["m_tot"]).any():
        a.plot(t, 100*(S["m_tot"]/S["m_tot"][0]-1), "-s", ms=3, color="tab:blue",
               label="total mass")
    a.axhline(0, color="0.6", lw=.8); a.set_xlabel("t [s]")
    a.set_ylabel("drift [%]"); a.set_title("1. Conservation"); a.grid(alpha=.3)
    a.legend(fontsize=8)

    a = ax[0][1]
    a.plot(t, S["R_V"]/R0, "-o", ms=3, color="tab:purple", label=r"$R_V$ (gas volume)")
    a.plot(t, S["R_05"]/R0, "-s", ms=3, color="tab:red", label=r"$R_{0.5}$ (radial avg)")
    if P is not None:
        a.plot(P[0], np.asarray(P[1])/R0, "--", color="k", lw=1.2, label="reference")
    a.set_xlabel("t [s]"); a.set_ylabel(r"$R/R_0$")
    a.set_title("2. Radius measures"); a.grid(alpha=.3); a.legend(fontsize=8)

    a = ax[1][0]
    if np.isfinite(S["band"]).any():
        a.plot(t, S["band"]/R0, "-o", ms=3, color="tab:green")
    a.set_xlabel("t [s]"); a.set_ylabel(r"band width / $R_0$")
    a.set_title("3. Interface band width"); a.grid(alpha=.3)

    a = ax[1][1]
    if S["p_probe"].ndim == 2:
        for j, name in enumerate(("origin", r"1.5$R_0$", r"2$R_0$", r"3$R_0$", "wall")):
            col = S["p_probe"][:, j]
            if np.isfinite(col).any():
                a.plot(t, col, "-", lw=1.4, label=name)
    a.set_xlabel("t [s]"); a.set_ylabel("pressure [Pa]")
    a.set_title("5. Pressure probes vs radius"); a.grid(alpha=.3); a.legend(fontsize=8)

    fig.suptitle(f"Driven bubble debug -- {stem}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    for e in ("png", "eps"):
        fig.savefig(os.path.join(IMG, f"driven_debug_{stem}.{e}"), dpi=DPI,
                    bbox_inches="tight")
    say(f"\n  wrote {os.path.join(IMG, f'driven_debug_{stem}.png')} / .eps")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--stem", default=None)
    a = ap.parse_args()

    kv = parse_input(a.input)
    R0 = num(kv, "eta.ic.expression.constant.R0", 0.02)
    p_inf = num(kv, "pressure0.ic.expression.constant.p_inf", 1.0e5)
    terms = drive_of(kv)
    stem = a.stem or os.path.basename(a.input).replace("input_", "")
    out = resolve_out(kv, a.output)

    say("="*78)
    say(f"DRIVEN BUBBLE DEBUG -- {stem}")
    say("="*78)
    say(f"  input      : {a.input}")
    say(f"  plotfiles  : {out or '(NOT FOUND)'}")
    say(f"  R0         : {R0:g} m     p_inf : {p_inf:g} Pa")
    for i, (A, w, p) in enumerate(terms):
        say(f"  drive[{i}]   : A = {A:g} Pa,  omega = {w:g} rad/s  (f = {w/2/math.pi:.4f} Hz)")
    if not out:
        say("\n  nothing to analyse."); return

    S = extract(out, R0, every=a.every)
    say(f"\n  frames read : {len(S['t'])}")
    if not len(S["t"]):
        return

    # optional reference curve from the existing comparison script
    P = None
    try:
        sys.path.insert(0, _HERE)
        os.environ.setdefault("CONFINE", "auto")
        import analyze_radius_drivencompare as M
        cfg = M.parse_input(a.input); Pm = M.build_params(cfg, a.input)
        tm, Rm = M.integrate("km", Pm, float(S["t"][-1]), 8000)
        P = (np.asarray(tm), np.asarray(Rm))
        say(f"  reference   : Keller-Miksis, CONFINE={os.environ.get('CONFINE')}, "
            f"f0_eff = {Pm['f0']:.3f} Hz")
    except Exception as exc:
        say(f"  [warn] reference model unavailable: {exc}")

    drift_table(S, R0)
    radius_table(S, R0)
    residual_table(S, R0, terms, P)
    probe_table(S, R0, terms, p_inf)
    energy_table(S)
    plots(S, R0, stem, P)

    np.savez_compressed(os.path.join(_HERE, f"driven_debug_{stem}.npz"),
                        R0=R0, p_inf=p_inf, terms=np.array(terms), **S)
    with open(os.path.join(_HERE, f"driven_debug_{stem}.txt"), "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    say(f"\n  wrote driven_debug_{stem}.npz  and  driven_debug_{stem}.txt")
    say("  >> transfer the .npz (small) rather than the plotfiles.")


if __name__ == "__main__":
    main()
