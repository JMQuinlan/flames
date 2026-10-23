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

Each run's overrides go in `OUT/<run>/cmdline.txt`.

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
- `input_Linear_ShellDamping_UNIT` at eps/R 0.25: 0.49 with the current code, 0.62 with contour-rate mode 1.
- The same at eps/R 0.125: 0.79 and 0.96.

## Planned: radially symmetric boundary condition (1D-in-3D slabs)

Goal: run spherical (3D) or cylindrical (2D) problems on a slab a few cells thick on the
desktop, e.g. a 3D bubble at R0/dx = 64 on 512 x 4 x 4 cells instead of an octant.

Idea:
- The domain is `x in [0, L]` and `|y|, |z| <= h` with h = 2–4 cells.
- On the y/z faces a new BC type, `RADIAL`, fills ghost cell (x, y_g, z_g) from the row
  nearest the axis at the same radius `r = sqrt(x^2 + y_g^2 + z_g^2)`:
  - scalars `f = f_axis(r)` by 1D interpolation;
  - vectors rotated, `u = u_r(r) (x, y_g, z_g)/r` (no swirl).
- `x = 0` keeps REFLECT.
- The Cartesian operators on the slab then see an exactly spherical field, so the
  geometric (D-1)/r terms come out of the ordinary 3D stencils. No new source terms.

Constraints and work:
- A box must span the slab in y/z. That is automatic for a slab a few cells thick.
- The x-interpolation reaches at most ~(y_g^2)/(2x) past the cell. It stays in the fab's
  ghost region except within a few cells of the axis.
- Implement in `BC::Constant` / `BC::Expression` as a new bcmap entry plus its fill.
  Momentum needs the vector rotation, which needs the component index.
- Verify with the uncoated Sch20 R(t) (slab vs octant) before using it for shell tests.
