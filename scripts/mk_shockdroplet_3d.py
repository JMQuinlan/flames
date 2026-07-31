#!/usr/bin/env python3
"""Generate 3D variants of the 2mm_WATER_2D shock-droplet inputs.

For each tests/FlowShockDroplet/2mm_WATER_2D/2mm_Droplet_*Ma file, write a
sibling <name>_3D with the 2D->3D conversion:
  * dim 2 -> 3
  * plot_file output dir gets a _3D suffix
  * domain + mesh extended in z (isotropic: dz = dx = dy)
  * droplet eta becomes a SPHERE: sqrt(x^2+y^2) -> sqrt(x^2+y^2+z^2)
  * per-phase velocity gets an explicit region2 (vz = 0)
  * momentum BCs become 3-component; z (front/back) slip walls added
The shock jump values (rho_shock, u_shock, p_shock) are preserved per file.
"""
import os
import re

SRC = os.path.join(os.path.dirname(__file__), "..",
                   "tests", "FlowShockDroplet", "2mm_WATER_2D")
DST = os.path.join(os.path.dirname(__file__), "..",
                   "tests", "FlowShockDroplet", "2mm_WATER_3D")
FILES = ["2mm_Droplet_1-5Ma", "2mm_Droplet_2-0Ma", "2mm_Droplet_3-0Ma",
         "2mm_Droplet_4-0Ma", "2mm_Droplet_5-0Ma"]

ZBC = (
    "\n# Front/back walls (z): Neumann (slip walls)\n"
    "density.bc.type.zlo = neumann\n"
    "density.bc.type.zhi = neumann\n"
    "energy.bc.type.zlo = neumann\n"
    "energy.bc.type.zhi = neumann\n"
    "momentum.bc.type.zlo = neumann neumann neumann\n"
    "momentum.bc.type.zhi = neumann neumann neumann\n"
)


def convert(txt):
    # 1. dimension
    txt = txt.replace("#@ dim=2", "#@ dim=3")
    # 2. plot_file output dir -> _3D
    txt = re.sub(r"(output_2mm_Droplet_[0-9]+-[0-9]+Ma)", r"\1_3D", txt)
    # 3. mesh: add z cells (isotropic; 0.01 m / dx = 128)
    txt = txt.replace("amr.n_cell = 256 128", "amr.n_cell = 256 128 128")
    # 4-5. domain: z spans the droplet symmetrically (-0.005 .. 0.005)
    txt = txt.replace("geometry.prob_lo = -0.005 -0.005 0.0",
                      "geometry.prob_lo = -0.005 -0.005 -0.005")
    txt = txt.replace("geometry.prob_hi =  0.015  0.005 0.0",
                      "geometry.prob_hi =  0.015  0.005  0.005")
    # 6. spherical droplet: add z_center + z^2 term
    txt = txt.replace(
        "eta.ic.expression.constant.y_center = 0.0  # Centered at origin",
        "eta.ic.expression.constant.y_center = 0.0  # Centered at origin\n"
        "eta.ic.expression.constant.z_center = 0.0  # Centered at origin")
    txt = txt.replace(
        "sqrt((x-x_center)**2 + (y-y_center)**2)",
        "sqrt((x-x_center)**2 + (y-y_center)**2 + (z-z_center)**2)")
    # 7. explicit z-velocity (0) for both phases
    txt = txt.replace(
        'velocity0.ic.expression.region1 = "0.0"',
        'velocity0.ic.expression.region1 = "0.0"\n'
        'velocity0.ic.expression.region2 = "0.0"')
    txt = txt.replace(
        'velocity1.ic.expression.region1 = "0.0"',
        'velocity1.ic.expression.region1 = "0.0"\n'
        'velocity1.ic.expression.region2 = "0.0"')
    # 8. momentum BCs -> 3-component (x/y faces); leave commented dirichlet alone
    txt = re.sub(r"(momentum\.bc\.type\.[xy][lh][io] = neumann neumann)$",
                 r"\1 neumann", txt, flags=re.MULTILINE)
    # 8b. add z-face BCs after the right-wall (xhi) momentum line
    txt = txt.replace(
        "momentum.bc.type.xhi = neumann neumann neumann\n",
        "momentum.bc.type.xhi = neumann neumann neumann\n" + ZBC, 1)
    return txt


def main():
    src = os.path.normpath(SRC)
    dst = os.path.normpath(DST)
    os.makedirs(dst, exist_ok=True)
    written = []
    for fn in FILES:
        with open(os.path.join(src, fn), "r") as f:
            txt = f.read()
        out = os.path.join(dst, fn + "_3D")
        with open(out, "w", newline="\n") as f:
            f.write(convert(txt))
        written.append(fn + "_3D")
    print(f"wrote {len(written)} 3D inputs to {dst}:")
    for n in written:
        print("  " + n)


if __name__ == "__main__":
    main()
