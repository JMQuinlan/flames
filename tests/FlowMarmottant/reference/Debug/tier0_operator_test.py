#!/usr/bin/env python3
"""Tier 0: is the discrete capillary force balanceable by ANY pressure field?

No solver involved.  We lay an analytic tanh circle on a uniform grid, build the
capillary force with the integrator's exact stencil chain, and ask a question
that does not depend on how the pressure is discretised:

    The exact capillary force on a circle is sigma*kappa*grad(eta), which is
    purely radial, hence IRROTATIONAL.  A pressure gradient is irrotational by
    construction.  So any SOLENOIDAL component in the discrete capillary force
    cannot be balanced by any pressure field whatsoever -- it is a rotational
    force with nothing to oppose it, and it must drive a vortical flow.

Helmholtz-decompose the discrete force (FFT; the force is compactly supported
near the interface so the periodic box is clean) and report

    xi = ||F_solenoidal|| / ||F||

xi = 0 means the operator is well balanced and the static bubble is a discrete
equilibrium.  xi = O(1) means it can never be, at any resolution or timestep.

Three operators are compared:
  wide     - what Hydro2.cpp does today: Numeric::Gradient (centred 2dx) to
             build Omega, then a centred 2dx divergence of Omega.
  compact  - the Tier-3 candidate: grad(eta) and Omega evaluated ON FACES with
             compact differences, then a conservative face-difference divergence
             matching the width of the HLLC pressure flux difference.
  csf      - sigma*kappa*grad(eta) with kappa from the code's formula, for
             reference (this is what div(Omega) collapses to for constant sigma).
"""
import os
import numpy as np

SIGMA = 3.41530864          # sigma_0 of the reference Laplace case
R0 = 0.020
L = 0.20                    # domain is [-L/2, L/2]^2, matching the input file
SMALL = 1.0e-10             # the code's |grad eta| cutoff


# ----------------------------------------------------------------- fields ---
def eta_field(N, eps):
    x = (np.arange(N) + 0.5) * (L / N) - L / 2
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.hypot(X, Y)
    return 0.5 * (1.0 + np.tanh((r - R0) / eps)), L / N, X, Y


def dc(f, ax, dx):
    """Centred 2dx difference -- Numeric::Gradient / Stencil.H:83."""
    return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2.0 * dx)


def omega_from_grad(gx, gy, sig):
    gem = np.hypot(gx, gy)
    live = gem >= SMALL                      # the code's early-return cutoff
    g = np.where(live, gem, 1.0)
    oxx = np.where(live, sig * (g - gx * gx / g), 0.0)
    oyy = np.where(live, sig * (g - gy * gy / g), 0.0)
    oxy = np.where(live, sig * (-gx * gy / g), 0.0)
    return oxx, oyy, oxy


# -------------------------------------------------------------- operators ---
def force_wide(eta, dx, sig):
    """Exactly the production chain: 2dx gradient -> Omega -> 2dx divergence."""
    gx, gy = dc(eta, 0, dx), dc(eta, 1, dx)
    oxx, oyy, oxy = omega_from_grad(gx, gy, sig)
    Fx = dc(oxx, 0, dx) + dc(oxy, 1, dx)
    Fy = dc(oxy, 0, dx) + dc(oyy, 1, dx)
    return Fx, Fy


def force_compact(eta, dx, sig):
    """Balanced-force candidate: Omega on faces, compact conservative divergence.

    On the x-face at i+1/2 the normal derivative is the 2-point difference and
    the tangential derivative is the 4-point average of the two straddling
    centred differences; likewise on the y-face.  The divergence is then a
    single-cell-wide face difference, the same width as the HLLC pressure flux
    difference it has to cancel.
    """
    # x-faces: index f means the face between cell f and f+1
    ex_n = (np.roll(eta, -1, 0) - eta) / dx
    ex_t = (np.roll(eta, -1, 1) + np.roll(np.roll(eta, -1, 0), -1, 1)
            - np.roll(eta, 1, 1) - np.roll(np.roll(eta, -1, 0), 1, 1)) / (4.0 * dx)
    xx_f, _, xy_f = omega_from_grad(ex_n, ex_t, sig)

    # y-faces: index f means the face between cell f and f+1 in y
    ey_t = (np.roll(eta, -1, 0) + np.roll(np.roll(eta, -1, 1), -1, 0)
            - np.roll(eta, 1, 0) - np.roll(np.roll(eta, -1, 1), 1, 0)) / (4.0 * dx)
    ey_n = (np.roll(eta, -1, 1) - eta) / dx
    _, yy_g, xy_g = omega_from_grad(ey_t, ey_n, sig)

    Fx = (xx_f - np.roll(xx_f, 1, 0)) / dx + (xy_g - np.roll(xy_g, 1, 1)) / dx
    Fy = (xy_f - np.roll(xy_f, 1, 0)) / dx + (yy_g - np.roll(yy_g, 1, 1)) / dx
    return Fx, Fy


def force_csf(eta, dx, sig):
    """sigma * kappa * grad(eta), kappa from the code's kappa_method=1 formula."""
    gx, gy = dc(eta, 0, dx), dc(eta, 1, dx)
    gem = np.hypot(gx, gy)
    lap = ((np.roll(eta, -1, 0) - 2 * eta + np.roll(eta, 1, 0))
           + (np.roll(eta, -1, 1) - 2 * eta + np.roll(eta, 1, 1))) / dx ** 2
    gmx, gmy = dc(gem, 0, dx), dc(gem, 1, dx)
    kap = -(lap / (gem + SMALL) - (gx * gmx + gy * gmy) / (gem ** 2 + SMALL))
    return sig * kap * gx, sig * kap * gy


# ------------------------------------------------------- Helmholtz split ----
def solenoidal_fraction(Fx, Fy, dx):
    """xi = ||F_perp|| / ||F||, the part no pressure field can cancel."""
    N = Fx.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    fx, fy = np.fft.fft2(Fx), np.fft.fft2(Fy)
    kdotf = KX * fx + KY * fy
    px, py = KX * kdotf / K2, KY * kdotf / K2      # curl-free part
    sx, sy = fx - px, fy - py                       # solenoidal remainder
    sx[0, 0] = sy[0, 0] = 0.0
    num = np.sqrt((np.abs(sx) ** 2 + np.abs(sy) ** 2).sum())
    den = np.sqrt((np.abs(fx) ** 2 + np.abs(fy) ** 2).sum())
    Sx, Sy = np.real(np.fft.ifft2(sx)), np.real(np.fft.ifft2(sy))
    return num / den, Sx, Sy


def azimuthal_peak(Sx, Sy, X, Y, nth=512):
    """Dominant azimuthal mode of the solenoidal force sampled on r = R0."""
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    N = X.shape[0]
    dx = L / N
    ii = np.clip(((R0 * np.cos(th) + L / 2) / dx - 0.5).round().astype(int), 0, N - 1)
    jj = np.clip(((R0 * np.sin(th) + L / 2) / dx - 0.5).round().astype(int), 0, N - 1)
    mag = np.hypot(Sx[ii, jj], Sy[ii, jj])
    amp = np.abs(np.fft.rfft(mag - mag.mean())) / nth
    return int(np.argmax(amp[1:]) + 1), amp


OPS = {"wide": force_wide, "compact": force_compact, "csf": force_csf}


def run(N, eps_over_dx):
    dx = L / N
    eps = eps_over_dx * dx
    eta, dx, X, Y = eta_field(N, eps)
    out = {}
    for name, fn in OPS.items():
        Fx, Fy = fn(eta, dx, SIGMA)
        xi, Sx, Sy = solenoidal_fraction(Fx, Fy, dx)
        n, _ = azimuthal_peak(Sx, Sy, X, Y)
        Fmag = np.hypot(Fx, Fy)
        out[name] = dict(xi=xi, mode=n, Fmax=Fmag.max(),
                         Smax=np.hypot(Sx, Sy).max())
    return out


MU = 0.15   # liquid viscosity of the sweep runs


if __name__ == "__main__":
    print(f"sigma={SIGMA}  R0={R0}  domain={L}  mu={MU}\n")
    print("xi   = ||F_sol|| / ||F||      fraction no pressure field can cancel")
    print("u_est= |F_sol|max * eps^2/mu  viscous-balance parasitic velocity scale")
    print("(measured |u|max in the production run was 0.489)\n")
    hdr = f"{'N':>5} {'eps/dx':>7} {'eps':>9} |"
    for nm in OPS:
        hdr += f" {nm+' xi':>11} {'|F|max':>10} {'u_est':>9} {'n':>3} |"
    print(hdr)
    print("-" * len(hdr))
    for N in (256, 512, 1024):
        for eod in (1, 2, 4, 8, 16):
            r = run(N, eod)
            eps = eod * L / N
            line = f"{N:5d} {eod:7d} {eps:9.2e} |"
            for nm in OPS:
                d = r[nm]
                u_est = d["Smax"] * eps * eps / MU
                line += (f" {d['xi']:11.3e} {d['Fmax']:10.2e}"
                         f" {u_est:9.2e} {d['mode']:3d} |")
            print(line + ("  <- production" if (N == 1024 and eod == 4) else ""))
        print()

    # ---------------------------------------------------------------- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, AXIS, GRID = "#1a1a1a", "#555555", "#dcdcdc"
    COL = {"wide": ("#0072B2", (None, None), "o", "wide (production)"),
           "compact": ("#D55E00", (5.5, 2.0), "s", "compact / balanced-force"),
           "csf": ("#009E73", (1.2, 1.8), "^", r"CSF  $\sigma\kappa\nabla\eta$")}
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 10,
        "axes.labelsize": 11, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "legend.fontsize": 9,
        "figure.facecolor": "white", "savefig.facecolor": "white"})

    eods = [1, 2, 4, 8, 16]
    res = {e: run(1024, e) for e in eods}
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.14, wspace=0.25)
    for a in (a0, a1):
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        a.set_xscale("log", base=2); a.set_yscale("log")
        a.set_xlabel(r"band resolution  $\varepsilon/\Delta x$")
        a.axvline(4, color=AXIS, lw=0.9, ls=":")

    for nm, (c, dash, mk, lab) in COL.items():
        a0.plot(eods, [res[e][nm]["xi"] for e in eods], lw=1.7, color=c,
                dashes=dash, marker=mk, ms=6, mec="white", mew=0.9, label=lab)
        a1.plot(eods, [res[e][nm]["Smax"] * (e * L / 1024) ** 2 / MU for e in eods],
                lw=1.7, color=c, dashes=dash, marker=mk, ms=6, mec="white",
                mew=0.9, label=lab)
    a0.set_ylabel(r"$\xi=\|F_{\mathrm{sol}}\|/\|F\|$")
    a0.set_title("(a)  unbalanceable fraction of the capillary force",
                 loc="left", fontsize=10.5)
    a0.legend(frameon=False, loc="upper right")
    a1.axhline(0.489, color=INK, lw=1.4, ls="-.")
    a1.text(1.05, 0.489, " measured $|u|_{max}$ = 0.489", va="bottom",
            fontsize=9, color=INK)
    a1.set_ylabel(r"predicted $|u|$ from the static residual")
    a1.set_title("(b)  static residual cannot explain the observed velocity",
                 loc="left", fontsize=10.5)
    for ext in ("pdf", "png"):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"tier0_operator.{ext}")
        fig.savefig(p, dpi=400 if ext == "png" else None)
        print("wrote", p)
