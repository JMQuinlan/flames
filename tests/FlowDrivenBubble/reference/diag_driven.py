# -*- coding: utf-8 -*-
"""
===============================================================================
FLOW-DRIVEN BUBBLE -- BLOWUP DIAGNOSTIC SWEEP
===============================================================================
Runs over every plotfile of a (possibly still-running / dying) driven-bubble
run and extracts the quantities that discriminate between the candidate
failure modes:

  H1  under-resolved eta band (IC epsilon too small for the finest dx)
        -> band width (cells) at t=0; input-deck echo of epsilon
  H2  gas-mass leak (boundary rectification / interior non-conservation)
        -> total per-phase mass ledgers m0(t), m1(t)
  H3  band smearing / halo diffusion (gas redistributing outward)
        -> radial band extents r(eta=0.1/0.5/0.9/0.99); R_vol vs R_eta split
  H4  grid-mode asphericity vs corner-drive anisotropy
        -> interface cubic invariant K (>0 axis-bulge = grid mode;
           <0 corner-bulge = drive anisotropy)
  H5  dt strangulation (what is blowing up, where)
        -> effective dt between frames (from plotfile step numbers);
           min p / max |u| and their locations

USAGE (INCLINE):
    python diag_driven.py <output_dir>              # e.g. .../output_LowAmp_NSCBC
    -> writes  <output_dir>/diag_driven.csv  (small -- copy this back)
    -> prints  a summary table + automatic verdict hints

Requires yt + numpy only.  Corrupt/partial plotfiles are skipped.  Frame
count is subsampled to <= MAX_FRAMES evenly (first/last always kept).
===============================================================================
"""
import os
import re
import sys
import numpy as np

MAX_FRAMES = 120
R0     = 0.02
CENTER = (0.0, 0.0, 0.0)
T_DRIVE = 1.0 / 81.545
BAND_LEVELS = (0.1, 0.5, 0.9, 0.99)


# ---------------------------------------------------------------------------
def echo_input_deck(out_dir):
    """Print the key lines of the archived input deck (alamo `metadata`)."""
    for cand in (os.path.join(out_dir, "metadata"),
                 os.path.join(os.path.dirname(out_dir.rstrip("/\\")), "metadata")):
        if os.path.isfile(cand):
            keys = ("epsilon", "max_level", "n_cell", "n_error_buf",
                    "drive_amp", "drive_omega", "sigma", "cfl", "stop_time",
                    "Limiter", "nghost", "refine_box", "eta.ic")
            print(f"--- input deck echo ({cand}) " + "-" * 20)
            with open(cand, errors="replace") as f:
                for ln in f:
                    if any(k in ln for k in keys) and not ln.strip().startswith("#"):
                        print("   ", ln.rstrip())
            print("-" * 60)
            return
    print("  [deck] no metadata file found next to the plotfiles")


def stepnum(d):
    m = re.match(r"(\d+)cell$", os.path.basename(d))
    return int(m.group(1)) if m else -1


def main(out_dir):
    import yt
    yt.funcs.mylog.setLevel(40)

    echo_input_deck(out_dir)

    pfs = sorted((os.path.join(out_dir, d) for d in os.listdir(out_dir)
                  if d.endswith("cell")
                  and os.path.isdir(os.path.join(out_dir, d))), key=stepnum)
    if len(pfs) > MAX_FRAMES:
        idx = sorted(set(np.linspace(0, len(pfs) - 1, MAX_FRAMES).astype(int)))
        pfs = [pfs[i] for i in idx]
    print(f"  sweeping {len(pfs)} plotfiles ...")

    rows = []
    prev_step, prev_t = None, None
    for pf in pfs:
        try:
            ds = yt.load(pf)
            ad = ds.all_data()
            t = float(ds.current_time)
            step = stepnum(pf)
            x = np.array(ad["x"]) - CENTER[0]
            y = np.array(ad["y"]) - CENTER[1]
            z = np.array(ad["z"]) - CENTER[2]
            r = np.sqrt(x * x + y * y + z * z)
            eta = np.array(ad["eta"])
            vol = np.array(ad["index", "cell_volume"])
            re0 = np.array(ad["rho_eta0"]); re1 = np.array(ad["rho_eta1"])
            p = np.array(ad["pressure"]); rho = np.array(ad["density"])
            vx = np.array(ad["velocityx"]); vy = np.array(ad["velocityy"])
            vz = np.array(ad["velocityz"])
            u = np.sqrt(vx * vx + vy * vy + vz * vz)

            # masses / volume radius
            m0 = float(np.sum(re0 * vol)); m1 = float(np.sum(re1 * vol))
            Vg = float(np.sum(np.clip(1.0 - eta, 0, 1) * vol)) * 8.0
            R_vol = (3.0 * Vg / (4.0 * np.pi)) ** (1.0 / 3.0)

            # radial profile -> band extents
            nb = 300
            edges = np.linspace(0.0, 2.0 * R0, nb + 1)
            mid = 0.5 * (edges[:-1] + edges[1:])
            wsum, _ = np.histogram(r, bins=edges, weights=vol)
            esum, _ = np.histogram(r, bins=edges, weights=eta * vol)
            with np.errstate(invalid="ignore", divide="ignore"):
                prof = esum / wsum
            ok = np.isfinite(prof)
            rp, ep = mid[ok], prof[ok]
            crossings = {}
            for lvl in BAND_LEVELS:
                cr = np.nan
                above = np.where(ep >= lvl)[0]
                if len(above):
                    i = above[0]
                    if i == 0:
                        cr = rp[0]
                    else:
                        f = (lvl - ep[i-1]) / (ep[i] - ep[i-1])
                        cr = rp[i-1] + f * (rp[i] - rp[i-1])
                crossings[lvl] = cr

            # asphericity: interface-weighted cubic invariant K
            # (g = (x^4+y^4+z^4)/r^4; sphere avg 3/5.  K>0 axis-bulge,
            #  K<0 corner-bulge)
            r4 = (x * x + y * y + z * z) ** 2
            msk = r4 > 0
            g = (x[msk]**4 + y[msk]**4 + z[msk]**4) / r4[msk]
            ag = np.clip(1.0 - eta, 0, 1)
            w = (ag * (1.0 - ag) * vol)[msk]
            K = float(np.sum(w * g) / np.sum(w) - 0.6) if np.sum(w) > 0 else np.nan

            # extremes + probes
            i_pmin = int(np.argmin(p)); i_umax = int(np.argmax(u))
            io = int(np.argmin(r))
            dt_eff = np.nan
            if prev_step is not None and step > prev_step:
                dt_eff = (t - prev_t) / (step - prev_step)
            prev_step, prev_t = step, t

            rows.append(dict(step=step, t=t, m0=m0, m1=m1,
                             R_vol=R_vol,
                             r010=crossings[0.1], r050=crossings[0.5],
                             r090=crossings[0.9], r099=crossings[0.99],
                             K=K, dt_eff=dt_eff,
                             p_origin=float(p[io]),
                             p_min=float(p[i_pmin]),
                             pmin_r=float(r[i_pmin]) / R0,
                             u_max=float(u[i_umax]),
                             umax_r=float(r[i_umax]) / R0))
        except Exception as exc:
            print(f"  [skip] {os.path.basename(pf)}: {exc}")
            continue

    if len(rows) < 2:
        print("  <2 usable frames -- nothing to report")
        return

    # ---- CSV ----------------------------------------------------------
    csv = os.path.join(out_dir, "diag_driven.csv")
    keys = list(rows[0].keys())
    with open(csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for rw in rows:
            f.write(",".join(f"{rw[k]:.8e}" if isinstance(rw[k], float)
                             else str(rw[k]) for k in keys) + "\n")
    print(f"\n  wrote {csv}  <-- copy this file back for analysis")

    # ---- summary table --------------------------------------------------
    print(f"\n{'t/T':>7} {'m1drift%':>9} {'R_vol/R0':>9} {'r05/R0':>7} "
          f"{'band(w/R0)':>10} {'K':>8} {'dt_eff':>10} {'p_orig':>10} {'u_max':>7}")
    m10 = rows[0]["m1"]
    for rw in rows[:: max(1, len(rows) // 20)] + [rows[-1]]:
        band = (rw["r099"] - rw["r010"]) / R0 if np.isfinite(rw["r099"]) else np.nan
        print(f"{rw['t']/T_DRIVE:7.3f} {(rw['m1']/m10-1)*100:+9.3f} "
              f"{rw['R_vol']/R0:9.4f} {rw['r050']/R0:7.3f} {band:10.3f} "
              f"{rw['K']:+8.4f} {rw['dt_eff']:10.3e} {rw['p_origin']:10.3e} "
              f"{rw['u_max']:7.2f}")

    # ---- automatic verdict hints ---------------------------------------
    print("\n--- verdict hints " + "-" * 42)
    band0 = (rows[0]["r099"] - rows[0]["r010"]) / R0
    print(f"  H1 band width at t=0: {band0:.3f} R0 "
          f"({'SUSPICIOUSLY THIN -- check epsilon vs finest dx'
          if band0 < 0.05 else 'plausible'})")
    m1d = (rows[-1]["m1"] / m10 - 1) * 100
    print(f"  H2 gas-mass drift over run: {m1d:+.2f}% "
          f"({'LEAKING' if abs(m1d) > 2 else 'conserved-ish'})")
    bandN = (rows[-1]["r099"] - rows[-1]["r010"]) / R0
    print(f"  H3 band width now: {bandN:.3f} R0 "
          f"({'SMEARED >3x initial' if bandN > 3 * band0 else 'held'})")
    Ks = [rw["K"] for rw in rows if np.isfinite(rw["K"])]
    if Ks:
        j = int(np.argmax(np.abs(Ks)))
        print(f"  H4 peak |K| = {abs(Ks[j]):.4f} "
              f"({'AXIS-bulge (grid mode)' if Ks[j] > 0 else 'CORNER-bulge (drive anisotropy)'}"
              f"; sphere ~ 2e-5)")
    dts = [rw["dt_eff"] for rw in rows if np.isfinite(rw["dt_eff"])]
    if len(dts) > 2:
        print(f"  H5 dt_eff: first {dts[0]:.3e} -> last {dts[-1]:.3e} "
              f"({'STRANGLING (x%.0f)' % (dts[0]/dts[-1]) if dts[-1] < dts[0]/3 else 'stable'})")
    print("-" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python diag_driven.py <plotfile_output_dir>")
        sys.exit(1)
    main(sys.argv[1])
