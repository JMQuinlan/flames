# RPE Benchmark Analysis Scripts

Standalone scripts comparing AMReX simulation output to Rayleigh-Plesset / Keller-Miksis ODE reference solutions.

## Layout

```
OtherTests/
├── input_RPE_01_MildExpansion         standalone test input
├── input_RPE_02_RayleighCollapse       standalone test input
├── input_RPE_03_LinearOscillation      standalone test input
├── input_RPE_04_StrongCollapse         standalone test input
├── citations.txt                       paper sources
└── reference/
    ├── rayleigh_plesset_solver.py     shared ODE module
    ├── analyze_01_MildExpansion.py
    ├── analyze_02_RayleighCollapse.py
    ├── analyze_03_LinearOscillation.py
    ├── analyze_04_StrongCollapse.py
    ├── README.md                       this file
    └── Images/                         output PNGs
```

## Run a single test

```bash
# from the repo root
./bin/hydro2-2d-g++ tests/FlowRayleighPlesset/OtherTests/input_RPE_01_MildExpansion
python tests/FlowRayleighPlesset/OtherTests/reference/analyze_01_MildExpansion.py
```

The analysis script:
1. Solves the Rayleigh-Plesset ODE with the same parameters as the input file.
2. Solves Keller-Miksis (compressibility-corrected) for comparison.
3. Walks the AMReX `output_*` directory and extracts R(t) by finding the eta=0.5 crossing along the +x axis.
4. Plots all three on one figure in `reference/Images/`.

If the AMReX output isn't there yet, the script just plots the analytical reference so you can see the expected shape.

## Notes

- All four tests are **independent**: pick one, run it, analyze it.
- Tests 02 and 03 share the same domain and resolution (the only differences are pressure conditions and stop time), so cached compiles are reused.
- The shared ODE solver `rayleigh_plesset_solver.py` is imported by every analysis script.
- The `Params` dataclass at the top of each analysis script must match its input file exactly. If you tune an input parameter, update the corresponding script.

## Reference equations (one ODE per script)

Each analysis script plots three reference curves:

1. **Chen 2D cylindrical RPE** (heavy blue line, **primary**) — the matching reference for a 2D Cartesian simulation, with logarithmic far-field cutoff. Same form as `rayleighplesset_UNIT_TEST.py` in this repo. A perfect 2D simulation should land on this curve.
2. **3D Rayleigh-Plesset** (light gray dashed) — for context only. Differs from 2D in gas exponent, curvature factor, and the absence of `ln(r_inf/R)` inertia.
3. **3D Keller-Miksis** (light gray dotted) — first-order liquid-compressibility correction to 3D RP. Most relevant for Test 04 where wall Mach number rises.

The `r_inf` parameter for Chen 2D is the far-field cutoff. For these tests, `r_inf = 5×R₀ = 5 mm` (half the simulation domain side), matching where the NSCBC4 BC enforces `p_inf`.

If your simulation R(t) lands on Chen 2D and *not* on RP/KM, that's the correct outcome — your 2D simulation is correctly modeling cylindrical (not spherical) dynamics.
