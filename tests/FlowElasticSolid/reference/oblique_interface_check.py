#!/usr/bin/env python3
"""
P wave obliquely incident on a bonded Al | Cu interface (tests/FlowElasticSolid/input_2D_ObliqueInterface).

  python3 oblique_interface_check.py <run_dir> [--outdir DIR]

Reference: plane-wave (Zoeppritz) coefficients for a welded interface.  Each wave has velocity
A p f(t - s.x), slowness s = d/c with the common s_y = sin(theta)/c_l1, and stress
sigma_ij = -[lam (p.s) delta_ij + mu (p_i s_j + p_j s_i)] A f;  continuity of v_x, v_y,
sigma_xx, sigma_xy at x = 0 gives the four outgoing velocity amplitudes.
Measured: for every x column the complex amplitude of the exp(i k_y y) Fourier mode of (u, v),
projected on each wave's polarisation; the packet peak is taken in the window where that
packet must be at the time of the plot file (group velocity c cos(theta) along x).
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
AL = dict(rho=2.7, gam=3.4, pinf=21.5, mu=26.0); CU = dict(rho=8.9, gam=4.22, pinf=34.2, mu=92.0)
P0, KX, KY, X0, W = 1.0e-4, 2, 1, -2.5, 0.4


def zoeppritz():
    for m in (AL, CU):
        m["cl"] = np.sqrt(m["gam"] * (P0 + m["pinf"]) / m["rho"] + 4 / 3 * m["mu"] / m["rho"]); m["ct"] = np.sqrt(m["mu"] / m["rho"])
        m["lam"] = m["rho"] * m["cl"]**2 - 2 * m["mu"]
    th = np.arctan2(KY, KX); sy = np.sin(th) / AL["cl"]
    def wave(m, c, sgn, shear):
        sn = sy * c; cs = np.sqrt(1 - sn * sn); d = np.array([sgn * cs, sn])
        p = np.array([-d[1], d[0]]) * (1 if sgn > 0 else -1) if shear else d
        s = d / c
        return dict(p=p, vx=c * cs * sgn, kx=2 * np.pi * KY * d[0] / d[1], row=np.array([p[0], p[1],
                    -(m["lam"] * p.dot(s) + 2 * m["mu"] * p[0] * s[0]), -m["mu"] * (p[0] * s[1] + p[1] * s[0])]))
    inc = wave(AL, AL["cl"], +1, False)
    out = {"reflected P": wave(AL, AL["cl"], -1, False), "reflected SV": wave(AL, AL["ct"], -1, True),
           "transmitted P": wave(CU, CU["cl"], +1, False), "transmitted SV": wave(CU, CU["ct"], +1, True)}
    names = list(out)
    M = np.column_stack([out[names[0]]["row"], out[names[1]]["row"], -out[names[2]]["row"], -out[names[3]]["row"]])
    A = np.linalg.solve(M, -inc["row"])
    for n, a in zip(names, A): out[n]["A"] = a
    return inc, out


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    inc, out = zoeppritz()
    pfs = sorted(glob.glob(os.path.join(d, "output", "*cell")))

    def demod(pf):
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        nx, ny = ds.domain_dimensions[:2]; lo, hi = float(ds.domain_left_edge[0]), float(ds.domain_right_edge[0])
        x = lo + (np.arange(nx) + 0.5) * (hi - lo) / nx; y = (np.arange(ny) + 0.5) / ny
        e = np.exp(-2j * np.pi * KY * y)[None, :]
        cu = 2 * (np.asarray(cg[("boxlib", "velocityx")])[:, :, 0] * e).mean(axis=1)
        cv = 2 * (np.asarray(cg[("boxlib", "velocityy")])[:, :, 0] * e).mean(axis=1)
        return float(ds.current_time), x, cu, cv
    t_hit = -X0 / inc["vx"]
    D = [demod(p) for p in pfs]
    # Numerical damping over the ~10 periods of travel is not negligible, so every packet's peak
    # amplitude is tracked in time and extrapolated (log-linear fit) to the instant t_hit at
    # which the packet centre is at the interface.
    def envelope(wv, x, cu, cv):
        # one wave type = one polarisation AND one x wavenumber: project on the polarisation, shift
        # its k_x to zero and low-pass (Gaussian, 0.6 wavelengths of the incident wave) -- the other
        # packets, which have a different k_x, are filtered out even where they overlap in space
        z = (wv["p"][0] * cu + wv["p"][1] * cv) * np.exp(-1j * wv["kx"] * x)
        sig = 0.6 / np.hypot(KX, KY) / (x[1] - x[0]); m = int(4 * sig)
        g = np.exp(-0.5 * (np.arange(-m, m + 1) / sig)**2); g /= g.sum()
        # the low-pass also lowers the peak of a Gaussian envelope of width w_w = W |vx_w| / vx_inc
        # by 1/sqrt(1 + sig^2/w_w^2): undo that (exact for a Gaussian packet)
        ww = W * abs(wv["vx"]) / inc["vx"]; sx = sig * (x[1] - x[0])
        return np.abs(np.convolve(z, g, mode="same")) * np.sqrt(1 + (sx / ww)**2)

    def track(wv, tlo, thi):
        pol, vx = wv["p"], wv["vx"]
        tt, aa = [], []
        for t, x, cu, cv in D:
            if not (tlo <= t <= thi): continue
            a = envelope(wv, x, cu, cv)
            xc = (X0 + vx * t) if vx == inc["vx"] and t < t_hit else vx * (t - t_hit)
            win = np.abs(x - xc) < 2.2 * W * abs(vx) / inc["vx"]
            tt.append(t); aa.append(a[win].max())
        c = np.polyfit(tt, np.log(aa), 1)
        return np.exp(np.polyval(c, t_hit)), -c[0], np.array(tt), np.array(aa)
    tsep = 2.6 * W / inc["vx"]                 # packet clear of the interface (and of its twin)
    Ainc, ginc, ti, ai = track(inc, tsep, t_hit - tsep)
    print(f"{d}: incident angle {np.degrees(np.arctan2(KY, KX)):.2f} deg; interface reached at t = {t_hit:.3f}")
    print(f"  incident P: amplitude at the interface {Ainc:.4e} (decay rate {ginc:.2f} /time, {100 * (1 - np.exp(-ginc / (AL['cl'] * np.hypot(KX, KY)))):.1f} % per period)")
    t1, x, cu, cv = D[-1]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    with open(os.path.join(outdir, "oblique_interface_summary.csv"), "w") as fo:
        fo.write("wave,measured_extrapolated,zoeppritz,error_pct,raw_at_end\n")
        for n, wv in out.items():
            tclear = t_hit + 2.6 * W / inc["vx"]
            Aw, g, tw, aw = track(wv, tclear, D[-1][0])
            meas, ref = Aw / Ainc, abs(wv["A"])
            print(f"  {n:15s} |amplitude| / incident = {meas:.4f}   Zoeppritz {ref:.4f}   ({100 * (meas / ref - 1):+.1f} %)   [raw at t = {t1:.2f}: {aw[-1] / Ainc:.4f}; decay {g:.2f} /time]")
            fo.write(f"{n},{meas:.5f},{ref:.5f},{100 * (meas / ref - 1):.2f},{aw[-1] / Ainc:.5f}\n")
            a = envelope(wv, x, cu, cv)
            xc = wv["vx"] * (t1 - t_hit); wid = 2.2 * W * abs(wv["vx"]) / inc["vx"]
            ax[1 if "SV" in n else 0].plot(x, a / Ainc, label=n); ax[1 if "SV" in n else 0].plot([xc - wid, xc + wid], [ref, ref], "k--", lw=0.8)
    for a_, tt in zip(ax, ("P polarisations", "SV polarisations")):
        a_.axvline(0, color="0.6", ls=":"); a_.set_ylabel("amplitude / incident"); a_.set_title(tt, fontsize=9); a_.legend(fontsize=8); a_.grid(alpha=0.3)
    ax[1].set_xlabel("x [m]   (Al | Cu interface at 0; dashed: Zoeppritz)"); ax[1].set_xlim(-4, 3.5)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "oblique_interface.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/oblique_interface.png")


if __name__ == "__main__":
    main()
