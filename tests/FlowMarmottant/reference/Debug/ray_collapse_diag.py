#!/usr/bin/env python3
"""Diagnose a local interface collapse (jet / dimple) in a 3D OCTANT bubble run.

Built for the INCLINE Light_Mili crash (11-03: eta=0.5 ray collapses at the first rebound,
then SubcycleShellViscous N=7e13).  Reads the plotfiles and the run log, writes a small
directory + tarball to copy back:

  OUT/summary.txt     per-frame table: R05 on the named rays, min/max over a ray sweep and the
                      direction of the min, raw eta range, band extremes (|u|, |grad eta|, rho, p,
                      nu_s proxy) with their locations
  OUT/frames.npz      the same per frame, plus the full R05(theta, phi) map of the ray sweep
  OUT/dump_<t>.npz    full finest-level block (eta, rho, p, u, v, w, rho_eta0, rho_eta1, shell,
                      grad_eta) for the last --ndump frames (float32, ~1-2 MB each)
  OUT/slices.png      eta / pressure / |u| on the three symmetry planes + the x=y plane, last frame
  OUT/rays.png        R05 vs t on every swept ray (min ray highlighted)
  OUT/log_dt.csv      step, time, level, level-local dt   (from --log)
  OUT/log_events.txt  warnings / errors / NaN lines and SubcycleShellViscous lines, with step

Usage (on the cluster, in the run's output directory's parent):
  python3 ray_collapse_diag.py OUTDIR --log slurm-XXXX.out [--tmin 3e-3] [--ndump 3] [--np 8]
  -> OUTDIR_raydiag/ and OUTDIR_raydiag.tgz
Needs numpy, scipy, matplotlib, yt.  The bubble is at the octant corner (0,0,0).
"""
import argparse, glob, os, re, sys, tarfile
import numpy as np

KAP_S = 0.72          # shell.kappa_s of the Light_Mili decks (nu_s proxy only)
RHO_FLOOR = 1.0       # shell.nus_rho_floor default


def frames(out):
    fl = glob.glob(os.path.join(out, "*cell"))
    return sorted(fl, key=lambda s: int(re.findall(r"(\d+)cell$", s)[-1]))


def ray_dirs(nt):
    """Directions over the octant: polar angle theta from +z, azimuth phi from +x, both 0..90 deg."""
    th = np.linspace(0, np.pi / 2, nt)
    ph = np.linspace(0, np.pi / 2, nt)
    return th, ph


NAMED = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1),
         "xy": (1, 1, 0), "xz": (1, 0, 1), "yz": (0, 1, 1), "xyz": (1, 1, 1)}


def r05(e, d, dx, L):
    from scipy.ndimage import map_coordinates
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    s = np.linspace(0, L * 0.98, 1500)
    pts = np.array([s * d[0], s * d[1], s * d[2]]) / dx - 0.5
    v = map_coordinates(e, pts, order=1, mode="nearest")
    i = np.where((v[:-1] - 0.5) * (v[1:] - 0.5) <= 0)[0]
    if not len(i):
        return np.nan
    i = i[0]
    return s[i] + (0.5 - v[i]) * (s[i + 1] - s[i]) / (v[i + 1] - v[i] + 1e-300)


def one_frame(a):
    f, L, nt, dump_dir = a
    import yt
    yt.funcs.mylog.setLevel(50)
    ds = yt.load(f)
    lev = ds.index.max_level
    dx = float(ds.index.get_smallest_dx())
    n = int(np.ceil(L / dx))
    cg = ds.covering_grid(lev, left_edge=ds.domain_left_edge, dims=[n, n, n])
    have = {fn for (_, fn) in ds.field_list}
    g = lambda s: cg["boxlib", s].d
    eraw = g("eta")
    e = np.clip(eraw, 0, 1)
    rho = g("density"); p = g("pressure")
    u, v, w = g("velocityx"), g("velocityy"), g("velocityz")
    um = np.sqrt(u * u + v * v + w * w)
    if "grad_etax" in have:
        ge = np.sqrt(g("grad_etax") ** 2 + g("grad_etay") ** 2 + g("grad_etaz") ** 2)
    else:
        ge = np.linalg.norm(np.array(np.gradient(eraw, dx)), axis=0)
    x = (np.arange(n) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    def loc(idx):
        i, j, k = np.unravel_index(idx, e.shape)
        return [float(X[i, j, k]), float(Y[i, j, k]), float(Z[i, j, k])]

    o = dict(t=float(ds.current_time), dx=dx, lev=lev, n=n, file=os.path.basename(f))
    o["eta_min"] = float(np.nanmin(eraw)); o["eta_max"] = float(np.nanmax(eraw))
    o["n_eta_out"] = int(np.sum((eraw < -1e-6) | (eraw > 1 + 1e-6)))
    o["n_nan"] = int(sum(np.sum(~np.isfinite(q)) for q in (eraw, rho, p, u, v, w)))
    o["Vgas"] = 8 * float(np.sum(1 - e)) * dx ** 3
    # ray sweep
    th, ph = ray_dirs(nt)
    M = np.full((nt, nt), np.nan)
    for a_, T in enumerate(th):
        for b_, P in enumerate(ph):
            M[a_, b_] = r05(e, (np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)), dx, L)
    o["R05map"] = M
    for k, d in NAMED.items():
        o["R05_" + k] = r05(e, d, dx, L)
    # non-monotone eta outside the main crossing on the cell rows ON the three axes and one row off
    # (a gas pocket in the liquid tail; 11-04: R4 showed it on the axis rows only, from 3.35 ms)
    for k, row in (("x", lambda q, m: q[:, m, m]), ("y", lambda q, m: q[m, :, m]), ("z", lambda q, m: q[m, m, :])):
        for m in (0, 1):
            v = row(eraw, m); i5 = int(np.argmax(v > 0.5)); rm = np.maximum.accumulate(v[i5:]); dd = rm - v[i5:]
            j = int(np.argmax(dd)); c = [0, 0, 0]; c["xyz".index(k)] = i5 + j
            for a_ in range(3):
                if a_ != "xyz".index(k): c[a_] = m
            o["dip_%s%d" % (k, m)] = float(dd[j]); o["dip_%s%d_r" % (k, m)] = (i5 + j + 0.5) * dx
            if m == 0:
                o["dip_%s_cell" % k] = dict(eta=float(eraw[tuple(c)]), p=float(p[tuple(c)]), rho=float(rho[tuple(c)]),
                                           rho_gas=float(g("rho_eta1")[tuple(c)] / max(1 - eraw[tuple(c)], 1e-30)) if "rho_eta1" in have else np.nan,
                                           rho_liq=float(g("rho_eta0")[tuple(c)] / max(eraw[tuple(c)], 1e-30)) if "rho_eta0" in have else np.nan)
    if np.isfinite(M).any():
        a_, b_ = np.unravel_index(np.nanargmin(M), M.shape)
        o["R05_min"] = float(M[a_, b_]); o["R05_min_theta"] = float(np.degrees(th[a_])); o["R05_min_phi"] = float(np.degrees(ph[b_]))
        o["R05_max"] = float(np.nanmax(M)); o["R05_mean"] = float(np.nanmean(M))
    # band extremes (1e-4 < eta < 1 - 1e-4) and whole-block
    band = (eraw > 1e-4) & (eraw < 1 - 1e-4)
    o["n_band"] = int(band.sum())
    def ext(name, q, mask, fn):
        qq = np.where(mask, q, np.nan)
        if not np.isfinite(qq).any():
            o[name] = np.nan; o[name + "_at"] = [np.nan] * 3; o[name + "_eta"] = np.nan; return
        idx = (np.nanargmax if fn == "max" else np.nanargmin)(qq)
        o[name] = float(qq.flat[idx]); o[name + "_at"] = loc(idx); o[name + "_eta"] = float(eraw.flat[idx])
    ext("u_max_band", um, band, "max")
    ext("u_max_all", um, np.ones_like(band), "max")
    ext("ge_max_band", ge, band, "max")
    ext("ge_max_all", ge, np.ones_like(band), "max")
    ext("rho_min_band", rho, band, "min")
    ext("p_min_band", p, band, "min")
    ext("p_max_band", p, band, "max")
    ext("p_min_all", p, np.ones_like(band), "min")
    ext("nus_max", KAP_S * ge / np.maximum(rho, RHO_FLOOR), band | (ge > 1e-10), "max")
    # radial velocity at the eta=0.5 surface
    b5 = (e > 0.45) & (e < 0.55)
    if b5.any():
        ur = (u * X + v * Y + w * Z) / np.maximum(R, 1e-30)
        o["ur05_mean"] = float(np.mean(ur[b5])); o["ur05_min"] = float(np.min(ur[b5])); o["ur05_max"] = float(np.max(ur[b5]))
    if dump_dir:
        F = dict(t=o["t"], dx=dx, eta=eraw, rho=rho, p=p, u=u, v=v, w=w, ge=ge)
        for s in ("rho_eta0", "rho_eta1", "shell", "energy0", "energy1"):
            if s in have:
                F[s] = g(s)
        np.savez_compressed(os.path.join(dump_dir, "dump_%.6fms.npz" % (o["t"] * 1e3)),
                            **{k: (np.asarray(val, np.float32) if np.ndim(val) else val) for k, val in F.items()})
    return o


def parse_log(logf, od):
    rx_start = re.compile(r"STEP (\d+) starts")
    rx_end = re.compile(r"STEP (\d+) ends\. TIME = ([0-9.eE+-]+)")
    rx_dt = re.compile(r"DynamicTimestep_Update\) lev=(\d+) ([0-9.eE+-]+) \(")
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    rows, ev, step, tprev = [], [], -1, 0.0
    with open(logf, errors="replace") as f:
        for line in f:
            line = ansi.sub("", line)
            m = rx_start.search(line)
            if m:
                step = int(m.group(1)); continue
            m = rx_dt.search(line)
            if m:
                rows.append((step, tprev, int(m.group(1)), float(m.group(2)))); continue
            m = rx_end.search(line)
            if m:
                tprev = float(m.group(2)); continue
            if re.search(r"WARNING|ERROR|Abort|abort|nan|NaN|inf\b|SubcycleShellViscous|CFL DIAG|dt_|c_max|vx_max|vy_max|vz_max", line):
                ev.append("step %d  %s" % (step, line.rstrip()[:400]))
    with open(os.path.join(od, "log_dt.csv"), "w") as f:
        f.write("step,time_at_step_start,lev,dt_level\n")
        for r in rows:
            f.write("%d,%.12e,%d,%.6e\n" % r)
    with open(os.path.join(od, "log_events.txt"), "w") as f:
        f.write("\n".join(ev[-20000:]) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--log", default=None, help="run log (slurm .out) to parse dt and warnings from")
    ap.add_argument("--R0", type=float, default=0.02)
    ap.add_argument("--box", type=float, default=1.25, help="analysis block edge in R0 (bubble <= 1 R0)")
    ap.add_argument("--tmin", type=float, default=0.0, help="skip frames before this time [s]")
    ap.add_argument("--ndump", type=int, default=3, help="dump full blocks for the last N frames")
    ap.add_argument("--nt", type=int, default=19, help="ray sweep resolution per angle (19 -> 5 deg)")
    ap.add_argument("--np", type=int, default=4)
    A = ap.parse_args()
    out = A.outdir.rstrip("/")
    od = out + "_raydiag"; os.makedirs(od, exist_ok=True)
    L = A.box * A.R0
    if A.log:
        rows = parse_log(A.log, od)
        print("log: %d dt rows, last step %s" % (len(rows), rows[-1][0] if rows else "-"))
    fl = frames(out)
    import yt
    yt.funcs.mylog.setLevel(50)
    keep = []
    for f in fl:
        try:   # AMReX plotfile Header: version, nvars, nvars names, dim, time
            with open(os.path.join(f, "Header")) as h:
                hl = [h.readline() for _ in range(300)]
            nv = int(hl[1]); t = float(hl[2 + nv + 1])
        except Exception:
            t = None
        if t is None or t >= A.tmin:
            keep.append(f)
    print("%d frames (of %d) to read" % (len(keep), len(fl)))
    jobs = [(f, L, A.nt, od if i >= len(keep) - A.ndump else None) for i, f in enumerate(keep)]
    if A.np > 1:
        from multiprocessing import Pool
        with Pool(A.np) as P:
            R = P.map(one_frame, jobs)
    else:
        R = [one_frame(j) for j in jobs]
    R.sort(key=lambda d: d["t"])
    np.savez(os.path.join(od, "frames.npz"), F=np.array(R, dtype=object))

    R0 = A.R0
    hdr = ("t[ms]   R05/R0: x      y      z      xy     xz     yz     xyz  | sweep min (th,ph)     max  "
           "| eta[min,max] nOut | u_band max @r/R0 eta | |ge|max  nu_s max | rho_band min | p_band min, max"
           " | axis-row eta dip x,y,z (row 1 off: x,y,z) @r/R0")
    lines = [hdr]
    for d in R:
        r = lambda q: np.linalg.norm(q) / R0
        lines.append(
            "%7.4f  " % (d["t"] * 1e3)
            + " ".join("%.4f" % (d["R05_" + k] / R0) for k in NAMED)
            + " | %.4f (%4.0f,%4.0f) %.4f" % (d.get("R05_min", np.nan) / R0, d.get("R05_min_theta", np.nan), d.get("R05_min_phi", np.nan), d.get("R05_max", np.nan) / R0)
            + " | [%.3g,%.3g] %d" % (d["eta_min"], d["eta_max"], d["n_eta_out"])
            + " | %.3g @%.3f %.2g" % (d["u_max_band"], r(d["u_max_band_at"]), d["u_max_band_eta"])
            + " | %.3g %.3g" % (d["ge_max_band"], d["nus_max"])
            + " | %.3g" % d["rho_min_band"]
            + " | %.4g, %.4g" % (d["p_min_band"], d["p_max_band"])
            + " | %.3f %.3f %.3f (%.3f %.3f %.3f) @%.3f" % (d["dip_x0"], d["dip_y0"], d["dip_z0"], d["dip_x1"], d["dip_y1"], d["dip_z1"], d["dip_x0_r"] / R0)
            + ("  NaN=%d" % d["n_nan"] if d["n_nan"] else ""))
    lines.append("")
    lines.append("locations of the extremes in the LAST frame (x, y, z in R0; eta there):")
    d = R[-1]
    for k in ("u_max_band", "u_max_all", "ge_max_band", "ge_max_all", "rho_min_band", "p_min_band", "p_max_band", "p_min_all", "nus_max"):
        a = np.array(d[k + "_at"]) / R0
        lines.append("  %-13s %11.4g  at (%.3f, %.3f, %.3f)  r=%.3f  eta=%.3g" % (k, d[k], a[0], a[1], a[2], np.linalg.norm(a), d[k + "_eta"]))
    lines.append("axis-row dip cells, last frame:")
    for k in "xyz":
        c = d["dip_%s_cell" % k]
        lines.append("  %s axis: dip %.3f at r=%.3f R0  eta %.4f  p %.4g  rho_gas %.4g  rho_liq %.5g" % (k, d["dip_%s0" % k], d["dip_%s0_r" % k] / R0, c["eta"], c["p"], c["rho_gas"], c["rho_liq"]))
    txt = "\n".join(lines)
    open(os.path.join(od, "summary.txt"), "w").write(txt + "\n")
    print(txt)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.array([d["t"] for d in R]) * 1e3
        M = np.array([d["R05map"] for d in R]) / R0
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for a_ in range(M.shape[1]):
            for b_ in range(M.shape[2]):
                ax[0].plot(t, M[:, a_, b_], color="0.7", lw=0.5)
        for k, c in (("x", "C0"), ("y", "C1"), ("z", "C2"), ("xyz", "C3")):
            ax[0].plot(t, [d["R05_" + k] / R0 for d in R], color=c, lw=1.5, label=k)
        ax[0].plot(t, [d.get("R05_min", np.nan) / R0 for d in R], "k--", lw=1.2, label="sweep min")
        ax[0].set_xlabel("t [ms]"); ax[0].set_ylabel("R05 / R0"); ax[0].legend(); ax[0].set_title("eta=0.5 crossing, every swept ray")
        sp = np.array([d.get("R05_max", np.nan) - d.get("R05_min", np.nan) for d in R]) / R0
        ax[1].semilogy(t, np.maximum(sp, 1e-6), label="R05 max-min (sweep) / R0")
        ax[1].semilogy(t, [d["u_max_band"] for d in R], label="max |u| band [m/s]")
        ax[1].semilogy(t, [d["nus_max"] for d in R], label="nu_s proxy max")
        ax[1].set_xlabel("t [ms]"); ax[1].legend(); ax[1].set_title("onset")
        fig.tight_layout(); fig.savefig(os.path.join(od, "rays.png"), dpi=130)
        dumps = sorted(glob.glob(os.path.join(od, "dump_*.npz")))
        if dumps:
            D = np.load(dumps[-1]); n = D["eta"].shape[0]; ext_ = [0, n * float(D["dx"]) / R0] * 2
            um = np.sqrt(D["u"] ** 2 + D["v"] ** 2 + D["w"] ** 2)
            planes = (("z=0 (xy plane)", lambda q: q[:, :, 0].T), ("y=0 (xz plane)", lambda q: q[:, 0, :].T),
                      ("x=0 (yz plane)", lambda q: q[0, :, :].T),
                      ("x=y diagonal plane", lambda q: np.array([q[i, i, :] for i in range(n)]).T))
            fig, ax = plt.subplots(3, 4, figsize=(17, 12))
            for c, (nm, sl) in enumerate(planes):
                for r_, (q, lab, cm) in enumerate(((D["eta"], "eta", "viridis"), (D["p"], "p", "RdBu_r"), (um, "|u|", "magma"))):
                    im = ax[r_, c].imshow(sl(q), origin="lower", extent=ext_, cmap=cm)
                    ax[r_, c].contour(sl(D["eta"]), [0.5], colors="w", linewidths=0.6, extent=ext_)
                    ax[r_, c].set_title("%s  %s  t=%.4f ms" % (lab, nm, float(D["t"]) * 1e3), fontsize=9)
                    plt.colorbar(im, ax=ax[r_, c], fraction=0.046)
            fig.tight_layout(); fig.savefig(os.path.join(od, "slices.png"), dpi=110)
    except Exception as ex:
        print("plotting failed:", ex)

    with tarfile.open(od + ".tgz", "w:gz") as tf:
        tf.add(od, arcname=os.path.basename(od))
    print("\nwrote %s/ and %s.tgz" % (od, od))


if __name__ == "__main__":
    main()
