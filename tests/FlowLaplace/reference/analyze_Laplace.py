# -*- coding: utf-8 -*-
"""
===============================================================================
FlowLaplace single-test analysis  --  2D Young-Laplace surface-tension test.
===============================================================================
For one R<R>_Sigma<sigma> test this script produces, from the AMReX output:

  1. R(t)/R0  along the x-axis ray (y = 0, eta = 0.5 crossing), with the
     y-axis ray overlaid as a symmetry check.
  2. The relative error |R/R0 - 1| on a semilog axis.
  3. An animated GIF of the eta = 0.5 interface contour over time.

A correct Brackbill CSF holds the bubble at R = R0 (p_gas - p_liquid = sigma/R,
2D single curvature), so R/R0 should stay flat at 1 and the error should stay
near round-off.  Drift / oscillation / collapse flags a surface-tension scaling
bug, or that this (R, sigma) is too stiff for the resolution.

USAGE:
    python tests/FlowLaplace/reference/analyze_Laplace.py R1.0e0_Sigma1.0e0
    python tests/FlowLaplace/reference/analyze_Laplace.py            # -> DEFAULT_TEST

OUTPUTS (in reference/Images/):
    <test>_radius.png / .eps      R/R0(t) and the error semilog
    <test>_interface.gif          eta = 0.5 contour animation
===============================================================================
"""

import os
import re
import sys
import fnmatch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

# ============================================================================
# ============================  CONFIGURATION  ===============================
# Everything tunable lives here -- no hunting through the body.
# ============================================================================

# Which test to analyze when none is given on the command line.
DEFAULT_TEST = "R1.0e0_Sigma1.0e0"

# Fluid constants (must match gen_inputs.py / the input files).
RHO_LIQ      = 0.991
P_LIQUID     = 1.0
EOS0_GAMMA   = 5.5
EOS0_P0      = 1.505
C_LIQ        = (EOS0_GAMMA * (P_LIQUID + EOS0_P0) / RHO_LIQ) ** 0.5

# Backup plotfiles/dirs matching this glob are ignored when reading output.
IGNORE_GLOB = "*.old.*"

# Analysis knobs.
ETA_THRESHOLD = 0.5          # interface iso-level
FRB_RES       = 400          # fixed-resolution buffer size for the gif frames
MAX_GIF_FRAMES = 60          # cap frames to keep the gif small

# ----- publish-quality plot style -----
STYLE = {
    "font.family":       "serif",
    "mathtext.fontset":  "cm",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   10,
    "axes.linewidth":    1.0,
    "lines.linewidth":   1.8,
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
}
FIGSIZE_RADIUS = (7.0, 6.5)
GIF_FPS        = 12
GIF_DPI        = 110

COLOR_X   = "#1f77b4"
COLOR_Y   = "#d62728"
COLOR_AVG = "#000000"
COLOR_ERR = "#2ca02c"
SAVE_EPS  = True
# ============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
IMG_DIR = os.path.join(_HERE, "Images")


def sci_latex(v, sig=3):
    """Format a number as LaTeX scientific notation, e.g. 1e-06 -> '10^{-6}',
    10.0 -> '10^{1}', 1.0 -> '10^{0}', 0 -> '0'.  Returned string is meant to
    sit inside a math-mode ($...$) context."""
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / 10.0 ** exp
    mant_str = f"{mant:.{sig}g}"
    if mant_str in ("1", "-1"):
        sign = "-" if mant_str == "-1" else ""
        return rf"{sign}10^{{{exp}}}"
    return rf"{mant_str}\times10^{{{exp}}}"


def parse_test_name(name):
    """R<R>_Sigma<sigma>  ->  (R0, sigma) as floats."""
    m = re.match(r"R(?P<R>[^_]+)_Sigma(?P<S>.+)", name)
    if not m:
        raise ValueError(f"cannot parse R0/sigma from test name '{name}'")
    return float(m.group("R")), float(m.group("S"))


def char_timescale(R0, sigma):
    """Capillary time if sigma > 0, else acoustic time R0/c_l."""
    if sigma > 0.0:
        return (RHO_LIQ * R0 ** 3 / sigma) ** 0.5
    return R0 / C_LIQ


def find_output_dir(name):
    """Locate the AMReX plot-directory for a test across the usual roots."""
    candidates = [
        os.path.join(_REPO, "tests", "FlowLaplace", f"output_{name}"),
        os.path.join(_REPO, "bin", "tests", "FlowLaplace", f"output_{name}"),
        os.path.join(os.getcwd(), "tests", "FlowLaplace", f"output_{name}"),
        os.path.join(os.getcwd(), "bin", "tests", "FlowLaplace", f"output_{name}"),
    ]
    for c in candidates:
        if os.path.isdir(c) and _cell_dirs(c):
            return c
    return None


def is_ignored(name):
    """True for backup artifacts (e.g. *.old.*) that should never be read."""
    return fnmatch.fnmatch(os.path.basename(name), IGNORE_GLOB)


def _cell_dirs(d):
    return sorted(
        os.path.join(d, x) for x in os.listdir(d)
        if os.path.isdir(os.path.join(d, x)) and x.endswith("cell")
        and not is_ignored(x)
    )


def extract_radius_history(output_dir, axis="x"):
    """Radius from the eta = ETA_THRESHOLD crossing along a ray from the origin.

    Returns (times, radii) sorted by time.  Mirrors the RPE reference
    extractor but is self-contained so FlowLaplace has no cross-test imports.
    """
    import yt
    yt.funcs.mylog.setLevel(40)

    times, radii = [], []
    for pf in _cell_dirs(output_dir):
        ds = yt.load(pf)
        L = float(ds.domain_right_edge[0])
        if axis == "x":
            ray = ds.ray(ds.arr([0.0, 0.0, 0.0], "code_length"),
                         ds.arr([L, 0.0, 0.0], "code_length"))
            s = np.array(ray["x"])
        else:
            ray = ds.ray(ds.arr([0.0, 0.0, 0.0], "code_length"),
                         ds.arr([0.0, L, 0.0], "code_length"))
            s = np.array(ray["y"])
        eta = np.array(ray["eta"])
        order = np.argsort(s)
        s, eta = s[order], eta[order]

        idx = np.where(eta >= ETA_THRESHOLD)[0]
        if len(idx) == 0:
            radii.append(np.nan)
        elif idx[0] == 0:
            radii.append(s[0])
        else:
            i = idx[0]
            frac = (ETA_THRESHOLD - eta[i - 1]) / (eta[i] - eta[i - 1])
            radii.append(s[i - 1] + frac * (s[i] - s[i - 1]))
        times.append(float(ds.current_time))

    t = np.array(times)
    o = np.argsort(t)
    return t[o], np.array(radii)[o]


def extract_eta_frames(output_dir):
    """Return (times, [eta_img...], extent) as uniform images for the gif."""
    import yt
    yt.funcs.mylog.setLevel(40)

    pfs = _cell_dirs(output_dir)
    if len(pfs) > MAX_GIF_FRAMES:
        step = int(np.ceil(len(pfs) / MAX_GIF_FRAMES))
        pfs = pfs[::step]

    times, imgs, extent = [], [], None
    for pf in pfs:
        ds = yt.load(pf)
        lo = ds.domain_left_edge
        hi = ds.domain_right_edge
        width = float(hi[0] - lo[0])
        slc = ds.slice(2, 0.0)
        frb = slc.to_frb((width, "code_length"), FRB_RES,
                         center=ds.arr([0.0, 0.0, 0.0], "code_length"))
        imgs.append(np.array(frb["eta"]))
        extent = [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]
        times.append(float(ds.current_time))
    return np.array(times), imgs, extent


def apply_style():
    plt.rcParams.update(STYLE)


def plot_radius(name, R0, sigma, t, Rx, Ry):
    apply_style()
    tau = char_timescale(R0, sigma)
    tn = t / tau                      # dimensionless time
    Ravg = 0.5 * (Rx + Ry)
    err = np.abs(Ravg / R0 - 1.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_RADIUS, sharex=True)

    ax1.axhline(1.0, color="0.4", lw=0.8, ls=":", label="equilibrium  $R=R_0$")
    ax1.plot(tn, Rx / R0,   color=COLOR_X,   ls="-",  label=r"$R_x/R_0$  ($y=0$)")
    ax1.plot(tn, Ry / R0,   color=COLOR_Y,   ls="--", lw=1.3, label=r"$R_y/R_0$  ($x=0$)")
    ax1.plot(tn, Ravg / R0, color=COLOR_AVG, ls="-",  lw=1.2, alpha=0.5, label=r"$\langle R\rangle/R_0$")
    ax1.set_ylabel(r"$R/R_0$")
    ax1.set_title(rf"Laplace test  $R_0={sci_latex(R0)}$,  $\sigma={sci_latex(sigma)}$  "
                  rf"($\Delta p=\sigma/R_0={sci_latex(sigma / R0 if R0 else 0)}$)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", framealpha=0.9)

    ax2.semilogy(tn, np.maximum(err, 1e-16), color=COLOR_ERR, ls="-",
                 label=r"$|\langle R\rangle/R_0 - 1|$")
    ax2.set_ylabel("relative error")
    ax2.set_xlabel(r"$t/\tau$" + rf"  ($\tau={tau:.3g}$)")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    png = os.path.join(IMG_DIR, f"{name}_radius.png")
    fig.savefig(png)
    if SAVE_EPS:
        fig.savefig(os.path.join(IMG_DIR, f"{name}_radius.eps"))
    plt.close(fig)
    print(f"  wrote {png}")
    return Ravg, err


def make_gif(name, R0, sigma, times, imgs, extent):
    if not imgs:
        print("  [warn] no frames for gif")
        return
    apply_style()
    tau = char_timescale(R0, sigma)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(imgs[0].T, origin="lower", extent=extent,
                   cmap="coolwarm", vmin=0.0, vmax=1.0, aspect="equal")
    cs = [ax.contour(np.linspace(extent[0], extent[1], imgs[0].shape[0]),
                     np.linspace(extent[2], extent[3], imgs[0].shape[1]),
                     imgs[0].T, levels=[ETA_THRESHOLD], colors="k", linewidths=1.5)]
    # reference circle at R0
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R0 * np.cos(th), R0 * np.sin(th), "w--", lw=1.0, alpha=0.7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\eta$")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    title = ax.set_title("")
    xs = np.linspace(extent[0], extent[1], imgs[0].shape[0])
    ys = np.linspace(extent[2], extent[3], imgs[0].shape[1])

    def update(k):
        im.set_data(imgs[k].T)
        cs[0].remove()                      # matplotlib >=3.10: ContourSet is one artist
        cs[0] = ax.contour(xs, ys, imgs[k].T, levels=[ETA_THRESHOLD],
                           colors="k", linewidths=1.5)
        title.set_text(rf"$R_0={sci_latex(R0)}$, $\sigma={sci_latex(sigma)}$   "
                       rf"$t/\tau={times[k] / tau:.2f}$")
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(imgs), blit=False)
    os.makedirs(IMG_DIR, exist_ok=True)
    gif = os.path.join(IMG_DIR, f"{name}_interface.gif")
    anim.save(gif, writer=animation.PillowWriter(fps=GIF_FPS), dpi=GIF_DPI)
    plt.close(fig)
    print(f"  wrote {gif}")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST
    print("=" * 70)
    print(f"FlowLaplace analysis: {name}")
    print("=" * 70)

    R0, sigma = parse_test_name(name)
    print(f"  R0 = {R0:g}, sigma = {sigma:g}, "
          f"sigma/R0 = {sigma / R0 if R0 else 0:g}, tau = {char_timescale(R0, sigma):.3g}")

    out = find_output_dir(name)
    if out is None:
        print(f"  [error] no AMReX output for '{name}'. Run the sim first, e.g.:")
        print(f"          ./bin/hydro2-2d-g++ tests/FlowLaplace/{name}")
        return 2
    print(f"  output dir = {out}")

    try:
        t, Rx = extract_radius_history(out, axis="x")
        _, Ry = extract_radius_history(out, axis="y")
    except Exception as e:
        print(f"  [error] radius extraction failed: {e}")
        return 2

    Ravg, err = plot_radius(name, R0, sigma, t, Rx, Ry)

    print(f"  frames        = {len(t)}")
    print(f"  R/R0 final    = {Ravg[-1] / R0:.6f}")
    print(f"  R/R0 min/max  = {np.nanmin(Ravg) / R0:.6f} / {np.nanmax(Ravg) / R0:.6f}")
    print(f"  max |R/R0-1|  = {np.nanmax(err) * 100:.4f} %")

    try:
        gt, imgs, extent = extract_eta_frames(out)
        make_gif(name, R0, sigma, gt, imgs, extent)
    except Exception as e:
        print(f"  [warn] gif generation failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
