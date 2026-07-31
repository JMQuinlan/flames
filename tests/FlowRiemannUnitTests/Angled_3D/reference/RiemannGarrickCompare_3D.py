"""
Overlay the 3D angled Garrick Riemann simulations against the exact
(stiffened-gas) 1D solution -- a multi-dimension coupling check.

The Garrick problem is 1D along the propagation direction n; its profile
projected onto n must be IDENTICAL for every orientation.  This script:

  1. Auto-discovers the runs by globbing ../input_Garrick_<PLANE>_<NN>
     (PLANE in {XY,YZ,XZ,XYZ}, NN the angle in deg) and keeping those whose
     output directory exists.
  2. For each, rebuilds the unit normal n(plane, theta), casts a yt ray along n
     through the origin, and projects:
         s   = n . r   = nx*x + ny*y + nz*z          (signed normal coordinate)
         u_n = v . n   = vx*nx + vy*ny + vz*nz        (normal velocity)
  3. Overlays density, u_n, pressure for all orientations on the exact solution
     (black), then prints L2/Linf error vs exact per orientation.  If the curves
     all collapse onto the exact line, x-y-z coupling is clean.

Run (after the sims):
    cd tests/FlowRiemannUnitTests/Angled_3D/reference
    python RiemannGarrickCompare_3D.py
"""
import glob
import math
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yt

yt.funcs.mylog.setLevel(40)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Propagation directions n(plane, theta) -- MUST match gen_angled_3d_inputs.py
# ---------------------------------------------------------------------------
_S2 = math.sqrt(2.0)
PLANES = {
    "XY":  lambda t: (math.cos(t), math.sin(t), 0.0),
    "YZ":  lambda t: (0.0, math.cos(t), math.sin(t)),
    "XZ":  lambda t: (math.cos(t), 0.0, math.sin(t)),
    "XYZ": lambda t: (math.cos(t), math.sin(t) / _S2, math.sin(t) / _S2),
}
PLANE_COLOR = {"XY": "tab:blue", "YZ": "tab:green",
               "XZ": "tab:orange", "XYZ": "tab:red"}

# ---------------------------------------------------------------------------
# Output location: INCLINE by default, auto-fall-back to the desktop tree so the
# same script works whether you analyze on /mmfs1 or after copying outputs local.
# ---------------------------------------------------------------------------
_INCLINE = "/mmfs1/home/ttryon/flames/bin/tests/FlowRiemannUnitTests/Angled_3D"
_DESKTOP = os.path.normpath(os.path.join(
    _SCRIPT_DIR, "..", "..", "..", "..", "bin",
    "tests", "FlowRiemannUnitTests", "Angled_3D"))
OUTPUT_BASE = _INCLINE if os.path.isdir(_INCLINE) else _DESKTOP

def output_dir(plane, nn):
    return os.path.join(OUTPUT_BASE, f"output_Garrick_Angled_3D_{plane}_{nn:02d}")

# Exact 1D solution (orientation independent) -- shared with the 2D test.
EXACT_HDF5 = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..",
                                           "reference", "Garrick.hdf5"))
EXACT_GROUP = "Garrick"

IMG_DIR  = os.path.join(_SCRIPT_DIR, "Images")
SAVE_NAME = "Garrick_Angled_3D_overlay"

_input_re = re.compile(r"^input_Garrick_([A-Z]+)_(\d+)$")


def last_plotfile(d):
    if not os.path.isdir(d):
        return None
    cells = sorted(os.path.join(d, c) for c in os.listdir(d)
                   if os.path.isdir(os.path.join(d, c)) and c.endswith("cell"))
    return cells[-1] if cells else None


def extract_profile(plotfile, n):
    """Ray along n through the origin -> (s, rho, u_n, p) sorted by s."""
    nx, ny, nz = n
    ds = yt.load(plotfile)
    smax = 0.99 / max(abs(nx), abs(ny), abs(nz))   # stay inside the unit cube
    ray = ds.ray(ds.arr([-smax * nx, -smax * ny, -smax * nz], "code_length"),
                 ds.arr([ smax * nx,  smax * ny,  smax * nz], "code_length"))
    x = np.array(ray["x"]); y = np.array(ray["y"]); z = np.array(ray["z"])
    vx = np.array(ray["velocityx"]); vy = np.array(ray["velocityy"])
    try:
        vz = np.array(ray["velocityz"])
    except Exception:
        vz = np.zeros_like(vx)
    rho = np.array(ray["density"]); p = np.array(ray["pressure"])
    s = x * nx + y * ny + z * nz
    u_n = vx * nx + vy * ny + vz * nz
    o = np.argsort(s)
    return s[o], rho[o], u_n[o], p[o], float(ds.current_time)


def main():
    print("=" * 64)
    print("3D ANGLED GARRICK -- coupling overlay vs exact   (base: %s)" % OUTPUT_BASE)
    print("=" * 64)

    # discover runs
    runs = []   # (plane, angle)
    for f in glob.glob(os.path.join(_SCRIPT_DIR, "..", "input_Garrick_*")):
        m = _input_re.match(os.path.basename(f))
        if not m:
            continue
        plane, nn = m.group(1), int(m.group(2))
        if plane in PLANES and os.path.isdir(output_dir(plane, nn)):
            runs.append((plane, nn))
    runs.sort(key=lambda pa: (list(PLANES).index(pa[0]), pa[1]))
    if not runs:
        raise SystemExit("No completed runs found under %s -- run the sims first "
                         "(RUN_ANGLED_3D.bash / RunAngled3D.sh)." % OUTPUT_BASE)
    print("Completed orientations: " +
          ", ".join(f"{p}_{a:02d}" for p, a in runs))

    # exact solution
    if not os.path.isfile(EXACT_HDF5):
        raise SystemExit("Exact solution not found: %s" % EXACT_HDF5)
    with h5py.File(EXACT_HDF5, "r") as h:
        g = h[EXACT_GROUP]
        s_ex = g["xvec"][:]; rho_ex = g["density"][:]
        u_ex = g["velocity"][:]; p_ex = g["pressure"][:]

    fig, ax = plt.subplots(3, 1, figsize=(12, 14))
    ekw = dict(color="k", lw=2.5, ls="-", label="Exact", zorder=10)
    ax[0].plot(s_ex, rho_ex, **ekw)
    ax[1].plot(s_ex, u_ex, **ekw)
    ax[2].plot(s_ex, p_ex, **ekw)

    # angle -> alpha (older/steeper angles fainter) for readability within a plane
    a_of = {a: 0.45 + 0.5 * i / 4 for i, a in enumerate([0, 30, 45, 60, 90])}

    metrics = []
    t_final = None
    for plane, nn in runs:
        pf = last_plotfile(output_dir(plane, nn))
        if pf is None:
            continue
        n = PLANES[plane](math.radians(nn))
        s, rho, u_n, p, t = extract_profile(pf, n)
        t_final = t_final or t
        c = PLANE_COLOR[plane]
        lbl = f"{plane} {nn:02d}°"
        ax[0].plot(s, rho, "--", color=c, lw=1.2, alpha=a_of.get(nn, 0.8), label=lbl)
        ax[1].plot(s, u_n, "--", color=c, lw=1.2, alpha=a_of.get(nn, 0.8), label=lbl)
        ax[2].plot(s, p,   "--", color=c, lw=1.2, alpha=a_of.get(nn, 0.8), label=lbl)
        metrics.append((plane, nn,
                        np.linalg.norm(rho - np.interp(s, s_ex, rho_ex)),
                        np.max(np.abs(rho - np.interp(s, s_ex, rho_ex))),
                        np.linalg.norm(u_n - np.interp(s, s_ex, u_ex)),
                        np.linalg.norm(p - np.interp(s, s_ex, p_ex))))
        print(f"  [ok] {plane}_{nn:02d}: {len(s)} pts, t={t:.5f}, "
              f"rho[{rho.min():.4f},{rho.max():.4f}]")

    ax[0].set_title("3D angled Garrick: exact (black) vs hydro2 over %d orientations, "
                    "t=%.4f s" % (len(runs), t_final or 0.0),
                    fontsize=14, fontweight="bold")
    ax[0].set_ylabel(r"Density"); ax[1].set_ylabel(r"Normal velocity $u_n=\mathbf{v}\cdot\mathbf{n}$")
    ax[2].set_ylabel(r"Pressure")
    for a in ax:
        a.set_xlabel(r"normal coordinate $s=\mathbf{n}\cdot\mathbf{r}$")
        a.grid(True, alpha=0.3); a.set_xlim([s_ex.min(), s_ex.max()])
    ax[0].legend(fontsize=8, ncol=4, loc="best")
    plt.tight_layout()

    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, SAVE_NAME)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".eps", bbox_inches="tight")
    print("\nOverlay saved: %s.png / .eps" % out)

    # error table -- consistency across orientations == clean coupling
    print("\n" + "=" * 64)
    print("ERROR vs EXACT per orientation (consistent values => clean coupling)")
    print("=" * 64)
    print(f"{'orient':>9s} {'rho L2':>12s} {'rho Linf':>12s} {'u_n L2':>12s} {'p L2':>12s}")
    for plane, nn, rl2, rli, ul2, pl2 in metrics:
        print(f"{plane+'_'+f'{nn:02d}':>9s} {rl2:>12.4e} {rli:>12.4e} {ul2:>12.4e} {pl2:>12.4e}")
    if metrics:
        rl2s = np.array([m[2] for m in metrics])
        print("-" * 64)
        print(f"rho L2 across orientations: min={rl2s.min():.4e} max={rl2s.max():.4e} "
              f"spread={rl2s.max()-rl2s.min():.4e}  (small spread = good coupling)")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
