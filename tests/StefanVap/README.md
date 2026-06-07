# Stefan-like vaporization verification

Isolated 1D checks that the phase-change mass coupling in `Hydro2` actually
moves mass from the liquid phase to the vapor phase. Motivated by the finding
that vaporization had ~no effect on the shock-droplet runs: the wiring is
correct, but the Spalding rate is many orders of magnitude too small over a
~100 µs shock interaction (diffusion-limited `Dv`, plus a driving force built
from the *local* cell mass fraction `Y≈rho_g/rho_l≈0.003` instead of a
saturation value).

## Stage 1 — prescribed constant rate (plumbing) — `ConstRate_1D`

Quiescent, no-shock, no-AMR, quasi-1D box: liquid slab (η≈0) | planar interface |
vapor (η≈1). A **known** interfacial mass flux is imposed via the new inputs
`apply_vap_const = 1`, `vap_const_mdot = <kg/m^2/s>`, bypassing the Spalding
closure (added in `Hydro2.cpp`/`.H`):

```
m_dot_Vap   = vap_const_mdot * |grad eta|
d(M_liq)/dt = -vap_const_mdot * INT|grad eta| dV = -vap_const_mdot * Ly
d(M_gas)/dt = +vap_const_mdot * Ly      ,   M_total = const
```

Run it, then check the auto-logged `integrals.dat`:

```
bin/hydro2-2d-... tests/StefanVap/ConstRate_1D     # Matt builds/runs
python3 tests/StefanVap/check_const_rate.py        # reads integrals.dat, prints PASS/CHECK
```

Pass criteria: `M_total` flat to machine precision (action-reaction),
`slope(M_liq) = -0.16 kg/s` (within a few %), and `d(M_gas) = -d(M_liq)`.

If this fails, the bug is in the source application (Hydro2.cpp:2180-2181
limiter-off, or 2565-2566 Pass B). If it passes, the plumbing is sound and the
problem is purely the Spalding closure → Stage 2.

## Bug found & fixed during Stage 1 setup

The first runs crawled (dt collapsed to ~1e-13 s, ~10^4x below the acoustic CFL).
Root cause was a wrong phase-change source for `eta`, shared by the Spalding,
cavitation, and const-rate blocks. The key fact is this branch's variable
convention (Hydro2.cpp:847-852):

```
rho      = eta*rho0 + (1-eta)*rho1     // single mixture density
rho_eta0 = rho * eta                   // NOT a true partial density (alpha_0*rho_0)
rho_eta1 = rho * (1-eta)
=> Y = rho_eta0/rho = eta               // eta is effectively the mass fraction
```

So the canonical 5-eq form `m_dot*(1/rho_g - 1/rho_l)` (which assumes true partial
densities) does NOT apply -- here `rho_eta0/eta = rho`, so it evaluates to ~0 and
then becomes ill-conditioned as the fields drift, driving eta to inconsistent
values, blowing up the EOS sound speed, and collapsing dt. The correct,
convention-consistent source keeps `eta = rho_eta0/rho` under the mass source
(d(rho_eta0)=+m_dot, d(rho)=0 by action-reaction):

```
eta_dot = d(rho_eta0/rho)/dt = m_dot / rho     // > 0 for evaporation, bounded
```

Fixed in `Hydro2.cpp` (all three phase-change blocks). This also matters for
Stage 2: the Spalding path had the same wrong source.

A second, separate bug: `is_periodic = 0 1 0` (periodic y) with BCs specified
only for x corrupted the ylo/yhi boundary rows on step 1 (rho_eta1 ~ 6511,
E ~ 6e9 in the boundary row -> huge local sound speed -> dt collapse). Fixed by
using `is_periodic = 0 0 0` with explicit neumann y-BCs (the proven droplet
setup); a y-uniform planar interface stays uniform under zero-gradient y-BCs.
Input-only change, no rebuild.

Notes:
- With `plot_int = 1` a full run (tens of thousands of steps once dt recovers)
  writes that many plotfiles -- raise it or rely on `plot_dt` once stable.
- This single-mixture-density model has no per-phase volume DOF, so phase change
  is a compositional shift at fixed mixture density; the pressure responds and
  drives a (physical) expansion. Keep `vap_const_mdot` moderate so that stays
  slow; reduce it if dt is still depressed after the fix.

## Stage 3a — inert carrier gas + vapor species (`species_transport = 1`)

A Spalding model only makes sense if the vapor diffuses through an inert
*carrier* gas, so Stage 3 switches the gas phase from "pure vapor" to
"carrier (air) + dodecane vapor". This is built incrementally; each checkpoint
has a dedicated input and is verified before the next.

Inputs (all `species_transport = 1`, `pp_flux_limiter = 0` — the FE path; a
parse guard enforces this until the Pass-B species flux is wired):

- **`Species_Scaffold_1D` (ckpt 1 — field scaffolding).** `Yv_init = 0.5`,
  no source, quiescent. Verifies only that the conserved vapor field `rho_vap`
  is allocated, initialized (`rho_vap = 0.5*rho_eta0`), and logged
  (`9:M_vapor`, `10:M_carrier`). Static — nothing should move.
- **`Species_Evap_1D` (ckpt 2 — vapor evolution + carrier conservation).**
  `Yv_init = 0`, prescribed const-rate evaporation. `rho_vap` advects on the
  mixture mass flux and gains `+m_dot`; since `rho_eta0` also gains `+m_dot`,
  the inert carrier (`= rho_eta0 - rho_vap`) is conserved. Here `eos0`'s cv/cp
  equal the vapor's, so the composition EOS (ckpt 3) is a **no-op** — this
  doubles as the 3a-2 regression.
- **`Species_AirCarrier_1D` (ckpt 3 / 3a-2 — composition-dependent gas EOS).**
  `eos0 = air` (γ=1.4), dodecane-vapor cv/cp supplied via `cv_vap`/`cp_vap`.
  The gas EOS is built per-cell as a frozen ideal-gas blend at the local vapor
  mass fraction `Yv = mass_frac_v = rho_vap/rho_eta0` (mass-weighted cv,cp;
  γ=cp/cv; p0=0), so the local gas γ drops from 1.4 toward ~1.023 as vapor
  accumulates near the interface.

Verify any of them with the shared windowed checker (early, pre acoustic-transit
window — after that the phase-change expansion advects mass out through the
walls, exactly as in Stage 1):

```
python3 tests/StefanVap/check_species.py <plot_file>/integrals.dat
```

Checks: `M_carrier` ~ constant (carrier conserved), `d(M_vapor) = -d(M_liq)`
(evaporated liquid becomes vapor), `M_vapor` grows from 0.

Known limitations (deferred, flagged in code): the **ghost-cell** primitives
loop and the **NSCBC** path still use the pure carrier `eos0` (valid only when
vapor does not reach a gas-side boundary — true for these neumann tests); the
**Pass-B** (limiter-on) species flux and multi-grid `rho_vap` ghost fill are not
implemented; and the **latent-heat** energy source for the injected vapor mass
(Stage 3c) is not added, so absolute gas temperature is not yet physical.

## Stage 3b — diffusion + conduction

A Spalding model only "does something" if the vapor can diffuse away from the
interface into the carrier (and, for the real Stefan problem, if heat conducts
in to supply the interface). Built incrementally on top of 3a.

- **`Species_Diffusion_1D` (ckpt 3b-1 — vapor species diffusion).** Turns on
  Fickian binary diffusion of vapor through the carrier,
  `d(rho_vap)/dt += div( rho_eta0 * Dv * grad(Y_v) )`, `Y_v = rho_vap/rho_eta0`
  (conservative central Laplacian, arithmetic-mean face `rho_eta0`, no-flux at
  the neumann walls). A matching parabolic limit `dt_species = cfl_v*dx^2/Dv` is
  added to the dt min. Gated on `Dv > 0`, so all the 3a tests (`Dv = 0`) are
  unchanged. Evaporation is left on so a vapor gradient is created at the
  interface; with `Dv > 0` it spreads into the carrier instead of piling up.
  `Dv = 2e-3` is exaggerated (~100x real) for visibility while keeping dt
  acoustic-limited. Verify with the same `check_species.py` (carrier conserved =
  diffusion is conservative; `d(M_vapor) = -d(M_liq)`; `M_vapor` grows); the
  spreading itself is the `mass_frac_v` profile broadening vs the `Dv = 0`
  control (`Species_AirCarrier_1D`).

  ```
  python3 tests/StefanVap/check_species.py /mmfs1/home/jquinlan/runs/stefan/output_species_diff/integrals.dat
  ```

- **`Conduction_1D` (ckpt 3b-2 — Fourier thermal conduction).** Adds
  `d(E)/dt += div( k grad T )` to the energy RHS, `k = eta*k0_thermal +
  (1-eta)*k1_thermal` (eta-blended like the viscosity; conservative central
  Laplacian, arithmetic-mean face `k`, clamped neighbors => adiabatic/no-flux at
  the neumann walls). A matching parabolic limit `dt_conduction =
  cfl_v*rho*cv*dx^2/k` (using `rho_min`, the smaller phase `cv`, the larger phase
  `k`) is added to the dt min. Gated on `k > 0`, so all conduction-off
  (`k0=k1=0`) runs — including every prior test and the droplet runs — are
  bit-for-bit unchanged. The test is a single-phase gas box (`eta = 1`),
  quiescent, at uniform pressure with a smooth `T` step imposed via a density
  profile (`T = p/(R rho)`): ~600 K | ~300 K. `k0_thermal = 2.6` is exaggerated
  (~100x real air) for visibility while keeping dt acoustic-limited. No phase
  change, no species — conduction is isolated. Verify conservation + stability
  with `check_conduction.py` (`E_total` constant => conduction conservative;
  `M_total` constant; finite); the conductive smoothing itself is the `T` profile
  relaxing vs the `k0_thermal = 0` control (rerun the same input with
  `k0_thermal = 0`).

  ```
  python3 tests/StefanVap/check_conduction.py /mmfs1/home/jquinlan/runs/stefan/output_conduction/integrals.dat
  ```

  Needed before the latent-heat sink (3c) gives a physical interface temperature.

## Stage 3c — latent heat of vaporization (`LatentHeat_1D`)

The Tammann/CPG EOS carries no formation/reference energy, so the phase-change
enthalpy is not in the internal energy and must be added explicitly. Evaporation
draws the latent heat from the mixture's sensible energy (evaporative cooling), so
the energy RHS gets a sink:

```
E_rhs += -L_vap * m_dot_Vap      [J/m^3/s] ,   m_dot_Vap > 0: liquid -> vapor
```

applied to the total `m_dot_Vap` (Spalding + const-rate + cavitation). Turned on
with `apply_latent_heat = 1` (default 0 = Phase-A mass-only behavior, so every
earlier run is unchanged). It flows through `Source[3]`, so both the FE path and
the Pass-B limiter path apply it (the limiter scales it with the other sources to
keep pressure admissible).

`LatentHeat_1D` makes the check **quantitative** by using the prescribed const
rate: over the early (pre-transit) window the energy and liquid-mass slopes share
the same `vap_const_mdot * INT|grad eta| dV` factor, so the ratio cancels the
discrete |grad eta| quadrature (the source of Stage 1's ~10% offset):

```
d(E_total) / d(M_liq) = L_vap          (exact, quadrature-free)
```

`check_latent.py` verifies `E_total` decreases (cooling), `d(E_total)/d(M_liq) ~
L_vap`, and `M_liq` decreases. Control: rerun with `apply_latent_heat = 0` → the
early-window `E_total` stays ~flat.

```
python3 tests/StefanVap/check_latent.py /mmfs1/home/jquinlan/runs/stefan/output_latent/integrals.dat
```

This is what makes an absolute interface temperature physical; combined with
conduction (3b-2) it sets up the quantitative Stefan / d²-law (3d) once the
saturation closure (Stage 2) replaces the prescribed rate.

## Stage 2 — physical Spalding driving force (`Saturation_1D`)

The legacy closure used the local cell mass fraction as the Spalding driving
force, but in this branch `Y = rho_eta0/rho = eta`, so the old
`B_M = (eta - Y_inf)/(1 - eta)` depends on the phase indicator `eta` **only** —
it is thermodynamically meaningless and **temperature-independent**. That (not
just the small magnitude) is why evaporation was a no-op. Stage 2 replaces it
with the real driving force, the Antoine **saturation** mass fraction `Y_s` at
the interface temperature:

```
p_sat(T) = 10^(A - B/(T+C)) bar        # same Antoine curve as HRM cavitation
x_s      = p_sat / p                   # Raoult/Dalton mole fraction
Y_s      = (x_s/R_v) / (x_s/R_v + (1-x_s)/R_g)   # mole->mass, R = cp - cv
B_M      = (Y_s - Y_inf) / (1 - Y_s)
```

The mole→mass conversion uses the specific gas constants `R = cp - cv` (the
universal constant cancels), so **no new molecular-weight inputs** are needed:
`R_v = cp_vap - cv_vap`, `R_g` from the carrier `eos0`. Implemented as the
`SaturationYs()` helper and turned on with `spalding_saturation = 1` (default 0
= legacy local-`Y`, so all earlier runs are unchanged).

Verification is the **temperature response**: a hotter interface gives a larger
`Y_s` (Antoine) and so a faster rate, while the legacy closure is flat in `T`.

**Interface temperature (band overshoot fixed by the mixing rule).** The driving
force uses the **interface** temperature `T = MixedTemperature(rho, p, eta)`, now the
**thermal-equilibrium** mixing rule

```
T = [ eta*(p+pi0)/((g0-1) cv0) + (1-eta)*(p+pi1)/((g1-1) cv1) ] / rho
```

(EOS.H). Each phase is a pure fluid at the shared `p` and a common `T` (intrinsic
`rho_k = (p+pi_k)/((g_k-1) cv_k T)`); inverting the volume-fraction density blend
`rho = eta*rho_0 + (1-eta)*rho_1` gives the form above. It uses **per-phase**
gamma/cv/pi, so `T` stays **bounded between the two pure-phase temperatures** across
the band — no overshoot. The old single-effective-fluid form
`(p+p0_eff)/((g_eff-1) rho cv_eff)` divided by the *collapsing* band density (which
falls toward the gas value while `p0_eff` lingers) and superheated the interface by
hundreds of K, clamping this closure. This is the single-density analog of the full
two-state model's per-phase thermal-T blend.

`Saturation_1D` is quiescent liquid|gas (carrier+vapor, `Dv` on, latent +
conduction **off** so `T` stays put). When both phases share a pure-T the band is
**isothermal** at it, so the test runs at **physical** temperatures, no clamping:

```
HOT  pure-T = 400 K  ->  band ~400 K   (rho_air=0.87108, rho_dod=683.706)
COLD pure-T = 300 K  ->  band ~300 K   (rho_air=1.16144, rho_dod=911.607)
rate_hot/rate_cold ~ 200  (band-integrated; clamped/legacy gives ~0.9)
```

`Yv_init = 0` so `M_vapor` starts at 0 and `M_vapor(t_win) ~ rate*t_win` is the
cleanest rate proxy. Run the cold and hot cases (swap the two density constants),
then:

```
python3 tests/StefanVap/check_saturation.py <cold>/integrals.dat <hot>/integrals.dat 300 400
```

`check_saturation.py` re-implements the closure (mixed-EOS `T(eta)` + Antoine +
Spalding) and integrates over the band to predict the ratio; it PASSES if the
observed `M_vapor` ratio is `> 3` and within ~2x of that prediction, and it warns
if any band cell exceeds 489 K. CONTROL: `spalding_saturation = 0` gives the
flat ~0.9 trend (legacy `B_M` tracks `eta`, not `T`).

**Production-density clamping — RESOLVED.** Two fixes together: the textbook-SG EOS
fix put the *bulk* liquid at a physical ~365 K (was ~860 K at 750 kg/m³), and the
thermal-equilibrium mixing rule removed the *diffuse-band* `T` overshoot (the
interface no longer superheats hundreds of K above the bulk). So the saturation
closure no longer clamps at production conditions. (Remaining design fork, Stage 3d:
`Y_inf` constant far-field vs local gas-side `mass_frac_v`.)

A design choice left for Stage 3d: `Y_inf` here is the constant far-field
`Y_infinity` (textbook Spalding). The resolved-Stefan alternative is the local
gas-side vapor mass fraction `mass_frac_v`, which self-limits as the local vapor
approaches saturation — the natural coupling once diffusion + conduction + latent
are all live.

## Stage 3d — quantitative Stefan / d²-law (TODO)

Turn saturation + latent heat + conduction on together and compare the interface
recession against the analytic Stefan solution (interface ~ sqrt(t)); likely
switch `Y_inf` to the local `mass_frac_v` for the self-consistent coupling.
