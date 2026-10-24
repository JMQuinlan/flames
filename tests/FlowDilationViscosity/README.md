# FlowDilationViscosity — unit tests for the shell dilatational viscosity

The Light deck's damping deficit traces to one term: the Boussinesq–Scriven
viscous shell stress `kappa_s (div_s u) P |grad eta|`. These tests exercise that
term alone, at several resolutions (R0/dx) and band thicknesses (eps/dx). Each test
has a sharp-interface reference.

| Deck | Fluids | What it isolates | Reference |
|---|---|---|---|
| `input_SinglePhase_Pulse_2D` | Garrick normalised liquid on **both** sides | The diffuse membrane operator only. Both phases are the same fluid, so the Kapila source `S = K div u` is zero and the eta contours move exactly with the fluid; there is no band-gas lag. | `reference/acoustic_shell_1d.py`: linear acoustics with a sharp dashpot membrane, jump `p_in - p_out = kappa_s (D-1)^2 u/R^2` |
| `input_Multiphase_Normalized_2D` | Garrick liquid + normalised gas (rho ratio ~1000) | Membrane plus a gas band (Kapila lag present) | Incompressible cylindrical Rayleigh–Plesset with `R_inf` = equal-area circle of the box |
| `input_Multiphase_WaterAir_2D` | Softened water + air, R0 = 2 um (Light scale) | As above, in the real-fluid units of the Light deck | Same cylindrical RP |

All three decks are 2D quarter domains with symmetry on `x = 0` and `y = 0`. They are
viscous-only: no elastic shell, `sigma = 0`, `mu = 0`. That makes `kappa_s` the only physical damping.
Run each with `kappa_s = 0` on the same mesh as well. The difference between the two runs
removes the mesh's own numerical damping.

## Running (desktop)

```
B=~/claude_scratch/b2d/bin/hydro2-2d-g++          # any 2D build
tests/FlowDilationViscosity/reference/run_sweep.sh $B OUT 1 4 pulse         # 12 runs
tests/FlowDilationViscosity/reference/run_sweep.sh $B OUT 1 4 norm          # 18 runs
tests/FlowDilationViscosity/reference/run_sweep.sh $B OUT 1 4 water quick   # 2 runs
```

Sweeps:
- pulse: R0/dx = 16, 32 (uniform mesh; 64 needs shell sub-cycling).
- multiphase: `amr.max_level` 1, 2, 3, i.e. R0/dx = 8, 16, 32.
- Each crossed with eps/dx = 1, 2, 4 and kappa on/off.

Each run's overrides go in `OUT/<run>.cmdline`.

## Analysis

```
reference/analyze_dilatation.py  OUT/P_N32_E2_K2 --baseline OUT/P_N32_E2_K0 --plot p.png
reference/analyze_cyl_damping.py OUT/norm_L2_E2_K1 --baseline OUT/norm_L2_E2_K0 \
                                 --input tests/FlowDilationViscosity/input_Multiphase_Normalized_2D
```

Both report `kappa_eff/kappa_s`, the value of the membrane viscosity that best fits the run.
1.0 is exact. The pulse script also reports:
- transmitted-peak and reflected-minimum ratios against the reference;
- the anisotropy between the axis and diagonal probes (Cartesian-mesh imprint).

Earlier 3D/2D results to compare against (FlowMarmottant):
- `input_Linear_ShellDamping_UNIT` at eps/R 0.25: 0.49 with the current code, 0.62 with the contour-rate term (removed 2026-10-24).
- The same at eps/R 0.125: 0.79 and 0.96.

## Radially symmetric slab BC (`RADIAL`, 2026-10-23)

`input_SinglePhase_Pulse_Slab2D` (cylindrical) and `input_SinglePhase_Pulse_Slab3D` (spherical)
run the pulse test as a 1D radial problem on a slab 4 cells thick:
`x in [0,L]`, `y (,z) in [0,h]`, centre at the origin.

- Set `RADIAL` on `yhi` (and `zhi` in 3D) for eta, density, energy and every momentum
  component. `xlo`, `ylo` (,`zlo`) must be symmetry planes (REFLECT), `prob_lo = 0`,
  `amr.max_level = 0`. No NSCBC or embedded solid. The code aborts otherwise.
- `Hydro2::RadialGhostFill` refills every ghost outside a RADIAL face from the near-axis
  row (j = k = 0) at the same radius `r = |x|`:
  - scalars by cubic interpolation, clipped to the bracketing samples;
  - momentum as `u_r(r) x/|x|`.
  The row is gathered over all ranks, so any box layout works.
- Cells on RADIAL faces use central stencils, as symmetry faces do.
- Analyse with `reference/analyze_dilatation.py RUN --slab --input DECK`.
  The reference uses D = the deck dimension. `anis` compares the axis row with the
  outermost slab row at the same radius.

Cost: 3D at R0/dx 64 is 384 x 4 x 4 cells.

