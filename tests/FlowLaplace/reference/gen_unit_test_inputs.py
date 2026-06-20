#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the FlowLaplace UNIT_TEST input set (static files, no runner).

Unlike TEMPLATES/RUN_LAPLACE.bash (which expands + launches mpirun itself), this
writes plain static input files so the SLURM driver alone controls parallelism
(srun --mpi=pmi2). One file per (R, sigma) is emitted as

    tests/FlowLaplace/UNIT_TEST_R<R>_Sigma<sigma>

with the dual desktop/INCLINE plot_file convention (INCLINE /mmfs1 active by
default, desktop commented) writing into a UNIT_TEST/ output subfolder.

All physics/derived quantities come straight from gen_inputs.make_input(), so
these stay bit-for-bit consistent with the committed R*_Sigma* matrix -- only
the plot_file line and the file name differ.

Matrix: 3 radii x 2 realistic air/water surface tensions = 6 cases
(sigma = 0 control dropped; add it back here if you want the no-ST baseline).

    python tests/FlowLaplace/reference/gen_unit_test_inputs.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import gen_inputs as gi  # noqa: E402  (reuse the exact formulas / template body)

TEST_DIR = os.path.normpath(os.path.join(_HERE, ".."))

R_VALUES     = [1.0e-3, 1.0e-4, 1.0e-5]   # 1 mm, 100 um, 10 um
SIGMA_VALUES = [0.036, 0.073]             # surfactant-laden, clean air/water


def main():
    written = []
    for R in R_VALUES:
        for sigma in SIGMA_VALUES:
            name, text = gi.make_input(R, sigma)      # e.g. R1.0e-3_Sigma3.6e-2
            base = f"output_{name}"

            old = f"plot_file = ./tests/FlowLaplace/output_{name}"
            dual = (
                f"# plot_file = ./tests/FlowLaplace/UNIT_TEST_2D/{base}"
                f"                                  # DESKTOP (run from bin/)\n"
                f"plot_file = /mmfs1/home/ttryon/flames/bin/tests/FlowLaplace/UNIT_TEST_2D/{base}"
                f"   # INCLINE /mmfs1 (default)"
            )
            if old not in text:
                raise RuntimeError(f"expected plot_file line not found for {name}")
            text = text.replace(old, dual)

            out = os.path.join(TEST_DIR, f"UNIT_TEST_2D_{name}")
            with open(out, "w", newline="\n") as f:
                f.write(text)
            written.append(os.path.basename(out))

    print(f"wrote {len(written)} Laplace UNIT_TEST inputs to {TEST_DIR}:")
    for n in written:
        print(f"  {n}")


if __name__ == "__main__":
    main()
