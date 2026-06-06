# FlowMarmottant — Marmottant (2005) coated-bubble surface tension tests

Validation suite for the diffuse-interface Marmottant model on the
`Hydro2_6Eqn_Marmottant` branch. Theory and design: `bin/Marmottant.md`.
Implementation: `apply_marmottant` + `marm.*` in `src/Integrator/Hydro2.{cpp,H}`.

The code reconstructs an effective radius from the local mean curvature,
`R_eff = geom_factor/|kappa|`, and feeds it through the three-regime
`sigma(R_eff)` (buckled → elastic → ruptured). **Requires `kappa_method = 1`.**

## Canonical parameters (identical across all inputs + scripts)

| param          | value   | meaning                                  |
|----------------|---------|------------------------------------------|
| `R0`           | 0.02    | initial bubble radius                    |
| `marm.chi`     | 14.56   | shell elastic modulus χ                  |
| `marm.R_buckling` | 0.018 | buckling radius (σ=0)                   |
| `marm.sigma_break`| 7.28  | rupture tension (scaled "water")        |
| `marm.geom_factor`| 1.0   | c_d: 2D-planar/cylinder = 1             |

Derived: `σ₀ = σ(R0) = 3.4153` (elastic), `R_rupt = 0.022045`,
2D Laplace jump `σ₀/R0 = 170.77`, balanced `p_gas = 670.77`.
So R0 sits on the elastic ramp, `R_buck/R0 = 0.90`, `R_rupt/R0 = 1.10`.

## Inputs

- **`input_Marmottant_Laplace_Static`** — bubble Laplace-balanced at rest on the
  elastic ramp; should hold `R = R0` forever. Validates the σ(R_eff)
  reconstruction and the σ_eff diagnostic field.
- **`input_Marmottant_OFF_regression`** — identical, but `apply_marmottant=0`
  with constant `sigma = σ₀ = 3.4153`. Same equilibrium; isolates any
  Marmottant-branch bug from the shared CSF machinery.
- **`input_Marmottant_Oscillating`** — gas driven below balance (`p_gas=400`) so
  the bubble compresses past `R_buck` into the buckled regime
  (min R/R0 ≈ 0.82, **compression-only**: upper turning point stays at R0).

## Reference / analysis (`reference/`)

- **`marmottant_model.py`** — σ(R_eff) mirroring the C++ exactly, R_rupt/σ₀
  helpers, and a coated Chen-2D RPE ODE. Run directly to print the operating
  point and draw the σ(R) curve. (Mirrors the code's spherical R² area law on
  purpose — the 2D cylinder area law is a deferred physics decision, see
  `bin/Marmottant.md` §6.)
- **`analyze_Marmottant_Laplace.py`** — radius drift, σ_eff-on-interface vs σ₀,
  and R_eff = geom/|κ| vs R0.
- **`analyze_Marmottant_Oscillating.py`** — R(t), σ(R(t)) trajectory on the
  curve, compression-only asymmetry metric, coated-RPE overlay.

All scripts run **before** any simulation (they draw the analytic references and
print the operating point); once plotfiles exist they auto-load them.

## Workflow

```bash
# from <repo>, build (WSL/Linux):
make hydro2-2d-g++
cd bin
# run a case (output -> bin/tests/FlowMarmottant/output_*):
./hydro2-2d-g++ ../tests/FlowMarmottant/input_Marmottant_Laplace_Static
# analyze (from <repo>):
cd ..
python tests/FlowMarmottant/reference/analyze_Marmottant_Laplace.py
```

Validation order (per `bin/Marmottant.md`): regression (OFF) → static Laplace
(reconstruction) → oscillating (physics).

> Note: `marm.geom_factor = 1.0` (2D-planar) is the deferred-geometry default;
> set `2.0` for a true axisymmetric sphere when the 2D/3D split lands. The
> reference model uses the same R² area law as the code, so these scripts test
> *implementation correctness*, not the 2D area-law physics.
