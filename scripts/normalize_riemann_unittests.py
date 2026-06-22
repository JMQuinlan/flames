#!/usr/bin/env python3
"""Normalize every FlowRiemannUnitTests input (top dir only) to RK3 + WENO3 + HLLC.

For each input_* / UNIT_TEST_* file directly in tests/FlowRiemannUnitTests:
  * Riemann_Solver = N  (numeral)         -> Riemann_Solver.type = hllc
  * Riemann_Solver.type = <anything else> -> Riemann_Solver.type = hllc
  * Limiter.type -> weno3                  (added after the Riemann line if absent)
  * integration.type -> RungeKutta         (uncommented/added)
  * integration.rk.type -> 3               (uncommented/added)
Prints a per-file change report.  Does NOT touch subdirectories or .py/.rst/test.
"""
import os
import re

DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                    "tests", "FlowRiemannUnitTests"))

RIEMANN = "Riemann_Solver.type = hllc"
LIMITER = "Limiter.type        = weno3"
INTTYPE = "integration.type    = RungeKutta"
INTRK   = "integration.rk.type = 3"

re_riem_num = re.compile(r"^\s*Riemann_Solver\s*=\s*\d")
re_riem_typ = re.compile(r"^\s*Riemann_Solver\.type\s*=")
re_lim      = re.compile(r"^\s*Limiter\.type\s*=")
re_it       = re.compile(r"^\s*#?\s*integration\.type\s*=")
re_rk       = re.compile(r"^\s*#?\s*integration\.rk\.type\s*=")


def main():
    files = sorted(f for f in os.listdir(DIR)
                   if os.path.isfile(os.path.join(DIR, f))
                   and (f.startswith("input_") or f.startswith("UNIT_TEST_")))
    for fn in files:
        path = os.path.join(DIR, fn)
        with open(path) as f:
            lines = f.read().split("\n")

        riemann_idx = None
        has_lim = has_it = has_rk = False
        changed = []
        out = []
        for ln in lines:
            if re_riem_num.match(ln) or re_riem_typ.match(ln):
                if ln.strip() != RIEMANN:
                    changed.append(f"Riemann '{ln.strip().split('#')[0].strip()}' -> hllc")
                out.append(RIEMANN)
                riemann_idx = len(out) - 1
                continue
            if re_lim.match(ln):
                has_lim = True
                if ln.strip() != LIMITER.strip().replace("  ", " "):
                    if "weno3" not in ln:
                        changed.append(f"Limiter '{ln.strip().split('=')[1].strip()}' -> weno3")
                out.append(LIMITER)
                continue
            if re_it.match(ln):
                has_it = True
                if ln.lstrip().startswith("#"):
                    changed.append("integration.type uncommented (RungeKutta)")
                out.append(INTTYPE)
                continue
            if re_rk.match(ln):
                has_rk = True
                val = ln.split("=")[-1].strip()
                if ln.lstrip().startswith("#") or val != "3":
                    changed.append(f"integration.rk.type -> 3 (was {'commented ' if ln.lstrip().startswith('#') else ''}{val})")
                out.append(INTRK)
                continue
            out.append(ln)

        if riemann_idx is None:
            print(f"{fn}: WARNING - no Riemann_Solver line found, SKIPPED")
            continue

        ins = []
        if not has_lim: ins.append(LIMITER)
        if not has_it:  ins.append(INTTYPE)
        if not has_rk:  ins.append(INTRK)
        if ins:
            changed.append("added: " + ", ".join(x.split("=")[0].strip() for x in ins))
            out[riemann_idx + 1:riemann_idx + 1] = ins

        with open(path, "w", newline="\n") as f:
            f.write("\n".join(out))
        print(f"{fn:32s}: {'; '.join(changed) if changed else 'already OK'}")


if __name__ == "__main__":
    main()
