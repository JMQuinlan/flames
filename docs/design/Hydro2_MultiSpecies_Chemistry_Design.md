# Hydro2: multi-species state and chemistry coupling — design document

Status: design investigation only. No source, deck or doc other than this file was changed.
Date: 2026-10-06. Branch: `Hydro2_6Eqn_DeformingSolid` (working tree, uncommitted edits present).

Conventions used below

- "phase" = one of the two immiscible constituents of the six-equation model (index `k`, code suffix `0`/`1`), separated by the volume fraction `eta` $\equiv \eta_1$.
- "species" = a chemical component inside a phase (index `i`), miscible with the other species of that phase.
- Line numbers refer to the working tree on 2026-10-06. `src/Integrator/Hydro2.cpp` and `Hydro2.H` were being edited by another session while this was written (the file grew by 7 lines during the investigation), so treat `Hydro2.cpp` line numbers as ±10 and use the function names as the stable anchor.
- **TODO verify** marks a statement that I could not confirm from the code on disk or from a source I actually opened during this investigation. Citations without that mark are ones I am confident of bibliographically; equation numbers inside papers were not re-read and are not quoted.

---

## 0. Summary

1. **Recommended model: keep two phases and one `eta`; make each phase a mixture of species.** The new evolved quantity is the phase-wise species partial density $m_{k,i} = \eta_k \rho_k Y_{k,i}$, one multi-component MultiFab with `ncomp = s = N_0 + N_1`. A substance that exists both as a liquid and as a vapour is two entries (one in each phase block), linked by the phase-change relaxation. This is the structure the existing `vapor` field already has (`Hydro2_PhaseChange.H:33-39`); the design generalises it.
2. **The Riemann solvers do not need to change.** Species ride on the phase mass flux that HLLC already returns (`Flux::mass0/mass1`), exactly as `vapor` does today (`Hydro2.cpp` ≈4083–4103). The per-phase EOS constants `gamma_k, pi_k` are already per-state metadata in `State`/`Primitive`; a species-dependent EOS enters by filling them per cell with a local stiffened-gas fit (Section 3.4).
3. **Mechanical (pressure) relaxation stays a scalar, monotone root solve** for any number of phases and species. A coupled Newton with an analytic Jacobian is needed for the *thermal* and *chemical-potential* relaxations (unknowns $p, T, \xi_1..\xi_M$) and inside the stiff chemistry integrator. Section 5 gives residuals and Jacobians.
4. **Chemistry: PelePhysics first, Mutation++ later for plasma.** A PelePhysics scaffold already exists in this repo (Makefile block, `PelePhysicsEOS.H`, `scripts/build_pele_deps.sh`, `src/pele_hello.cc`) but the libraries themselves are **not on disk**. Before either library, a small native backend (constant-$\gamma$ species, one-step Arrhenius) gives tests with closed-form answers.
5. **Size of the two-phase hard-coding:** 1331 non-comment source lines in 25 files (609 in `Hydro2.cpp`). The recommended design touches roughly a third of them; converting to N volume fractions would touch all of them.

---

## 1. Current state

### 1.1 Evolved state and fields

The solver carries the Saurel–Petitpas–Berry (2009) six-equation set in the Schmidmayer–Bryngelson–Colonius (2020) form with a redundant total energy:

$$\mathbf{U} = \big(\eta,\ (\eta\rho)_1,\ (\eta\rho)_2,\ \rho\mathbf{u},\ \rho E,\ E_1,\ E_2\big),\qquad E_k \equiv (\eta\rho e)_k .$$

| Quantity | Field (`Hydro2.H`) | Registration (`Hydro2.cpp`, in `Parse`) | ncomp | plot | evolving |
|---|---|---|---|---|---|
| $\eta \equiv \eta_1$ | `eta_mf`, `eta_old_mf` (330, 339) | 874–875, BC `eta_bc` | 1 | `eta` | yes |
| $(\eta\rho)_k$ | `rho_eta0_mf`, `rho_eta1_mf` + `_old` (332–335) | 876–879, BC `density0_bc`/`density1_bc` | 1 each | `rho_eta0`, `rho_eta1` | yes |
| $\rho\mathbf{u}$ | `momentum_mf` (280) | 993 | D | `momentum` | yes |
| $\rho E$ | `energy_per_vol_mf` (282) | 989 | 1 | `energy_per_vol` | yes |
| $E_k$ | `energy0_mf`, `energy1_mf` (243, 262) | 943, 964 | 1 each | `energy0`, `energy1` | yes |
| shell $\Gamma$ | `shell_mf` (343) | 884 | 1 | `shell` | yes |
| colour fn (dead) | `cfun_mf` (381) | 887 | 1 | diag | yes |
| vapour $m_\text{vap}$ | `phasechange.vapor_mf` (`Hydro2_PhaseChange.H:69`) | 897–899, only if `phasechange.on` | 1 | `vapor`, `pc_T`, `pc_Tsat`, `pc_Yv`, `pc_dm` | yes |
| cobasis | `elastic.ebasis_mf` | 917–918, only if `elastic.on` | `NEB` | `ebasis` | yes |
| derived | `density_mf`, `velocity_mf`, `pressure_mf`, `T_mf`, `pressure0/1_mf`, `velocity0/1_mf`, `density0/1_mf`, `momentum0/1_mf` … | 939–1033 | — | mixed | mostly no |

`RegisterNewFab(field, bc, ncomp, nghost, name, writeout, evolving, suffix)` is declared at `Integrator.H:219`. `evolving = true` means the field is `FillPatch`ed on regrid (`Integrator.cpp:1222`) with `amrex::cell_cons_interp` (`Integrator.cpp:436, 477`). Every cell fab is averaged down fine→coarse after the fine sub-cycle (`Integrator.cpp` ≈1250–1255). Multi-component fields with per-component plot suffixes are already supported and used (`hess_eta`, `ebasis`, `pc`).

Ghost width is `nghost = 2` (`Hydro2.H:754`); WENO5 needs 3 and the code warns (`Hydro2.cpp` ≈1125).

### 1.2 Time integration wiring (`Hydro2::Advance`, ≈4738)

- Old/new pointer swaps for every primary: ≈4749–4762.
- The RK solution vector is a list of aliased MultiFabs with **fixed positional indices**: `[0] rho_eta0, [1] rho_eta1, [2] momentum, [3] rho E, [4] eta, [5] energy0, [6] energy1, [7] shell, [8] cfun`, then optional `[ix_el]` cobasis and `[ix_pc]` vapour (`ix_el = 9`, `ix_pc = 9 + elastic.on`, ≈4763; vectors built ≈4779–4802).
- `RHS(...)` takes each of these as a separate named argument (`Hydro2.H:138-161`; definition ≈1721).
- Post-stage hook (≈4847–4891): copy stage → working fields, clamp `eta` to [0,1], `FillGhost4BC`, copy back. `FillGhost4BC` (≈6427) calls `RelaxAndReinit(lev)` and then, if on, `RelaxPhaseChange(lev)` (≈6745–6748). So relaxation runs after every RK stage unless `defer_relax` is set.

### 1.3 Face state, reconstruction, Riemann solve (`Hydro2::RHS`)

- A per-tile primitive cache (`NPC = 4 + D (+1 in 2D)` components, ≈2477–2512) stores $(\eta, \rho_1, \rho_2, p_\text{mix}, \mathbf{u})$. It mirrors `Limiter::ToPrimitive` and hard-codes the Tammann inversion $p_k = (\gamma_k-1)E_k/\eta_k - \gamma_k\pi_k$ with `eos0`/`eos1` constants.
- `get_prim` (≈2513–2541) builds a `Limiter::Primitive` and sets `gamma0, pi0, gamma1, pi1` from `eos0`, `eos1` (≈2539–2540).
- `compute_face` (≈2810–2944): 6-cell window → `limiter->Reconstruct` (L and R) → `Limiter::ToState` → optional elastic face state → `riemannsolver->Solve(sL, sR, pref, small)` (≈2940) → `alpha_face` upwinded on `u_interface`.
- `Solver::Local::Limiter::Primitive` (`Limiter.H:65-79`): `alpha, alpha_rho_0, alpha_rho_1` (holding *pure* densities — see the comment at `Limiter.H:130-162`), `u, v, w, p_mix`, and metadata `gamma0, pi0, gamma1, pi1` described as "constant across the stencil and not reconstructed". `ToPrimitive` (`Limiter.H:120-184`) and `ToState` (`Limiter.H:191-221`) hard-code two phases and the Tammann form.
- `Solver::Local::FluidRiemann::State` (`Riemann.H:33-121`): `alpha, alpha_rho_0, alpha_rho_1, M_normal, M_tangent, M_tangent2, E0, E1, E_total`, per-phase `gamma0, pi0, gamma1, pi1` ("not advected; copied per call", `Riemann.H:54`), plus elastic members. `Flux` (`Riemann.H:130-166`): `mass0, mass1, momentum_*, energy_total, energy0, energy1, u_interface, alpha_face`, elastic tangential velocities.
- Flux divergence: mass/momentum/energy rows are conservative; the `eta` row is $-\nabla\!\cdot(\eta u^*) + \eta\,\nabla\!\cdot u^*$ built from `u_interface` (≈3900–3924). Per-phase pressures for the non-conservative $E_k$ rows use `EOS::PhasicPressureFromEnergy(..., eos0.Gamma(), eos0.P0(), ...)` (≈3739–3740).
- The limiter and Riemann solver are called through **virtual functions on host pointers** captured by value into the `ParallelFor` lambda (`limiter->Reconstruct`, `riemannsolver->Solve`). This compiles for CPU/OpenMP only.

### 1.4 EOS (`src/Solver/EOS/`)

- `EOS.H`: abstract `EOS` with `Gamma(), P0(), Cv(), Cp(), Pressure(rho,e), Temperature(rho,e), SoundSpeed(rho,p), InternalEnergy(rho,p)` (80–133). All mixture helpers are **static functions of exactly two EOS objects or two $(\gamma,\pi)$ pairs**: `MixedGamma/MixedP0/MixedCv/MixedCp` (144–199), `MixedPressure`, `MixedInternalEnergy`, `MixedTemperature` (272–284), Wood `MixedSoundSpeed` (329–365), `KapilaK` (386–411), `PhasicPressureFromEnergy` (454–472), `PhasicEnergyFromPressure` (478–492), `PhasicSoundSpeed` (498–513), `FrozenMixtureSoundSpeed(Y0,Y1,c0,c1)` (521–527), `InterfacialPressureZ` (534–542), `PhasicHugoniotPressure` (552–570), `RelaxationVolume_SG_SC` and its derivative (621–650), `ReinitMixturePressure(rho_e, alpha0, alpha1, gam0, pi0, gam1, pi1)` (687–700).
- `Tammann.H`: $p = (\gamma-1)\rho e - \gamma p_0$, $T = (p+\gamma p_0)/((\gamma-1)\rho c_v)$, $a = \sqrt{\gamma(p+p_0)/\rho}$; parsed from `eos0.*`, `eos1.*` (`Hydro2.cpp:439-440`). Both phases are `Solver::EOS::Tammann` objects (`Hydro2.H:736-737`); a gas is a Tammann fluid with `p0 = 0`.
- `CPG.H`: calorically perfect gas, same interface; not used by Hydro2 (both phases are `Tammann`).
- There is **no formation enthalpy or energy offset** in the Hydro2 EOS. The phase-change module adds $q, q'$ on its own side (next subsection).
- **Two temperature conventions coexist.** The plotted `T` uses the Tammann mixture convention `MixedTemperature` (≈1610, 6723, 7342); phase change uses $T = (p+\pi)/((\gamma-1)c_v\rho)$ (`Hydro2_PhaseChange.H:26, 77-78`), and the file says so explicitly (`:29-31`). Any species/chemistry work must fix one definition of the phase temperature.
- **EOS backend switch.** `EOS.H:21-76` defines `enum class Backend {Native, PelePhysics}` selected by `eos.backend` (`Hydro2.cpp:449`). Under `#ifdef PELE_ENABLED` the three `Phasic*` helpers route the gas branch (`pi_k == 0`) to `PelePhysicsEOS` (`EOS.H:460-509`). `PelePhysicsEOS.H` (197 lines) wraps only `GammaLaw` with one species and reproduces the native formulae; its header lists "Phase 3: multi-species, Phase 4: chemistry" as TODO (`PelePhysicsEOS.H:29-36`). The dispatch reads a host-side static inside kernels (`EOS.H:30-35` notes this is CPU-only).

### 1.5 Pressure relaxation (`Hydro2::RelaxAndReinit`, ≈8129–8674)

Per cell, skipping embedded-solid cells:

1. Clamp $\eta$ to `[1e-12, 1-1e-12]`; pure densities $\rho_k^0 = (\eta\rho)_k/\eta_k$; pre-relax pressures $p_k^0$ from $E_k$ with the Tammann inversion and the positivity floor $p_k + \pi_k > 0$.
2. **Pure-cell guard** (≈8322–8385): if $\eta$ is beyond `cutoff` or either $p_k^0$ sits at its floor, skip the solve, freeze $\eta$, and only reinitialise $E_k$ from $\rho E$.
3. **Scalar Newton** on $f(p) = \sum_{k=1}^{2}(\eta\rho)_k\,v_k(p) - 1$ (≈8390–8470) with the self-consistent closed form (`RelaxationVolume_SG_SC`)
$$v_k(p) = v_k^0\,\frac{p_k^0 + \gamma_k\pi_k + (\gamma_k-1)p}{\gamma_k(p+\pi_k)},\qquad \frac{dv_k}{dp} = -\frac{v_k^0(p_k^0+\pi_k)}{\gamma_k(p+\pi_k)^2} < 0 .$$
   Initial guess $p = \eta_1 p_1^0 + \eta_2 p_2^0$ (the dissertation text states the impedance-weighted $p_I$; the code uses the $\eta$-weighted mixture pressure). Steps are clamped into a bracket; `max_iter = 30`, tolerance `1e-10`.
4. **Bisection fallback** (≈8470–8510) on the same monotone $f$, up to 120 halvings, with bracket expansion.
5. $\eta_k^\text{new} = (\eta\rho)_k v_k(p^*)$, renormalised; if $|f| > 10^{-6}$ the old $\eta$ is kept (≈8536).
6. **Reinit**: $p_\text{reinit}$ from the conserved $\rho e = \rho E - \tfrac12\rho|\mathbf u|^2 - w_\text{el}$ via `ReinitMixturePressure` (≈8572), then $E_k \leftarrow \eta_k^\text{new}(p_\text{reinit}+\gamma_k\pi_k)/(\gamma_k-1)$.

Everything is closed-form in two $(\gamma,\pi)$ pairs. The loop over `k` is written out by hand (`gam0/gam1`, `pi0_/pi1_`, `arh0_loc/arh1_loc`, `v0/v1`).

### 1.6 Phase change (`Hydro2_PhaseChange.H`, `Hydro2::RelaxPhaseChange` ≈8675–8728)

- Default off (`phasechange.on = 0`). One liquid phase (`phasechange.liquid` = 0 or 1) and one gas phase.
- Stiffened gas with energy and entropy constants: `struct Phase { g, pi, cv, q, qp }` (`:58`); `vol(p,T)`, `ener(p,T)`, `temp(p,rho)`, `gibbs(p,T)` (`:73-80`). $q, q'$ come from `data/hydro2_materials.dat` or the deck.
- **One extra conserved scalar already exists**: `vapor` $= (\eta\rho)_\text{gas} Y_v$ — the partial density of the condensable species inside the gas phase. Air and vapour share one gas EOS and equal molar masses are assumed (`:33-39`). It is advected with flux = gas-phase mass flux × first-order upwind $Y_v$ (`Hydro2.cpp` ≈4083–4103).
- Cell solve `relax(...)` (`:145-215`): unknown = vapour mass $x$; for each trial $x$ an inner `ptEquilibrium` (`:108-139`) finds $(p,T)$ from volume and energy (secant, then bisection); the outer loop walks/doubles from the current composition until the Gibbs residual $g_l(p,T) - g_v(x_v p,T)$ changes sign, then Illinois regula falsi. Bounds on $x$ replace the Gibbs equality when one side is exhausted.
- Energy convention (`:48-49`): "The hydrodynamic phase energies of Hydro2 exclude q, so the conserved rho E changes by $-(\Delta m_\text{vap})(q_\text{vap}-q_\text{liq})$". `RelaxPhaseChange` writes $E_k$ from the equilibrium $p$ and adds the difference to $\rho E$ (≈8718).
- **Finding: `vapor` is not refluxed.** The flux register has `2 + D + 1 + 2` components (`Hydro2.cpp` ≈8742) and `CrseInit`/`FineAdd` cover mass, momentum, total energy and $E_k$ only (≈5019–5029). The `cc_fluxes` struct has no slot for it (`Hydro2.H:771-776`). With sub-cycled AMR the vapour mass is therefore not conserved across coarse–fine boundaries. A species implementation must add its fluxes to the register; this also repairs `vapor`.

### 1.7 AMR, reflux, average-down

- `AllocateRefluxScratch` (≈8729): `FluxRegister` with `ncomp_reflux = 2 + D + 1 + 2`; `cc_fluxes[lev].mass[d]` has 2 components, `ene_k[d]` 2 components.
- Stage-weighted accumulation of face fluxes into `cc_fluxes` in `RHS` (≈3840–3900), with `mass0/mass1/energy0/energy1` written out by name.
- `PostSubcycleReflux` (≈8782): six explicit `Reflux` calls with named component indices.
- `PostAverageDown` (≈8845–8935): on coarse cells under fine cells, re-derives `density`, `velocity` and overwrites $E_k$ from $\rho E$ using the two-phase closed form.
- `Regrid` (≈6044): reallocates reflux scratch, re-inits `phi`, calls `FillGhost4BC`.

### 1.8 Boundary and initial conditions

- BC objects: `density_bc`, `density0_bc`, `density1_bc`, `momentum_bc`, `energy_bc`, `pressure_bc`, `eta_bc`, `temperature_bc` (`Hydro2.H:410-425`). Per-phase density BCs exist because a Dirichlet mixture density would be applied to both partial densities (`Hydro2.cpp` ≈776–797, comment at `Hydro2.H:411-418`).
- `FillGhost4BC` (≈6427–7896) is the largest two-phase block after `RHS`: primitive recovery (≈6654), per-phase BC fill (≈7047–7048), primitive-BC path with EOS reconstruction of $E_k$ (≈7088), NSCBC path (≈6943). `vapor` is filled with `eta_bc` (≈6753).
- NSCBC: `src/BC/NSCBC.H` (995 lines) and `NSCBC4.H` (2597 lines) take `eos0`/`eos1` and use mixture $\Gamma,\Pi$.
- ICs: `density0_ic, velocity0_ic, pressure0_ic` and the `1` set (`Hydro2.H:431-441`); `Mix(lev)` (≈1377–1672) builds the conserved state. `vapor` is initialised as `Yv0 ×` gas partial density (≈1352–1360).

### 1.9 Count of two-phase hard-codings

Method: non-comment source lines matching any of five token classes — mass (`rho_eta0/1`, `alpha_rho_0/1`, `mass0/1`, `density0/1_`, `arh0/1`), energy (`E0/E1`, `E0_/E1_`, `energy0/1`, `ene_k`), EOS constants (`eos0/1`, `gamma0/1`, `pi0/1`, `p0_0/1`, `cv0/1`, `cp0/1`, `mu0/1`, `MW0/1`, `gam0/1`), per-phase primitives (`pressure0/1_`, `velocity0/1_`, `momentum0/1_`, `rho0/1`, `c0/1`, `Z0/1`, `Y0/1`), and complements (`1.0 - eta`, `1.0 - a1`, …). The script is `bin/tests/species_scratch/count2phase.sh` (read-only; re-run it after edits). Classes overlap; "union" counts each line once.

| File | mass | energy | EOS | prim | compl. | **union** |
|---|---|---|---|---|---|---|
| `src/Integrator/Hydro2.cpp` | 265 | 172 | 100 | 98 | 23 | **609** |
| `src/Integrator/Hydro2.H` | 21 | 9 | 6 | 16 | 2 | 54 |
| `src/Integrator/Hydro2_PhaseChange.H` | 0 | 0 | 4 | 0 | 0 | 4 |
| `src/Integrator/Hydro2_Solid.H` | 0 | 0 | 4 | 5 | 2 | 9 |
| `src/Integrator/EmbeddedSolid.H` | 1 | 1 | 0 | 0 | 0 | 1 |
| `src/Solver/EOS/EOS.H` | 0 | 0 | 68 | 9 | 10 | 79 |
| `FluidRiemann/HLLC.H` | 17 | 21 | 8 | 10 | 1 | 45 |
| `FluidRiemann/HLLCE.H` | 27 | 23 | 19 | 10 | 1 | 70 |
| `FluidRiemann/HLLC_All_Mach.H` | 21 | 23 | 12 | 8 | 1 | 57 |
| `FluidRiemann/HLLC_All_Mach_Furfaro.H` | 27 | 23 | 12 | 8 | 1 | 61 |
| `FluidRiemann/HLLC_Elastic.H` | 16 | 18 | 8 | 10 | 1 | 41 |
| `FluidRiemann/HLLC_Oomar_Jaiman.H` | 27 | 23 | 12 | 8 | 1 | 61 |
| `FluidRiemann/HLLE.H` | 14 | 9 | 4 | 8 | 1 | 26 |
| `FluidRiemann/Lax_Friedrich.H` | 14 | 9 | 4 | 7 | 1 | 25 |
| `FluidRiemann/Riemann.H` | 12 | 10 | 14 | 0 | 0 | 36 |
| `FluidRiemann/Roe.H` | 16 | 11 | 11 | 8 | 1 | 37 |
| `FluidRiemann/Upwind.H` | 9 | 7 | 2 | 0 | 1 | 15 |
| `Limiter/Limiter.H` | 13 | 5 | 19 | 0 | 2 | 35 |
| `Limiter/{MUSCL,Minmod,VanLeer,WENO3,WENO5}.H` | 2 each | 0 | 2 each | 0 | 0 | 4 each (20) |
| `src/BC/NSCBC.H` | 0 | 0 | 8 | 0 | 5 | 13 |
| `src/BC/NSCBC4.H` | 0 | 0 | 19 | 8 | 6 | 33 |
| **Total (25 files)** | | | | | | **1331** |

Grouped by what a refactor would have to do:

| Group | Lines | Needed for option (a), N volume fractions | Needed for option (b), recommended |
|---|---|---|---|
| Riemann solvers + `Riemann.H` (11 files) | 474 | rewrite all | **none** (species are passive; $\gamma,\pi$ metadata already per state) |
| Limiters (6 files) | 55 | rewrite `Primitive`, `ToPrimitive`, `ToState`, operators | none in the limiters; `get_prim` fills metadata from a per-cell cache |
| `EOS.H` | 79 | all two-argument mixture helpers → loops | add species thermodynamics beside it; helpers keep their signatures |
| `Hydro2.cpp` + `Hydro2.H` | 663 | essentially all | the ~100 EOS-constant lines in `Hydro2.cpp` (replace `eosK.Gamma()/P0()` by per-cell values when `species.on`), plus additive code for the new field |
| NSCBC | 46 | rewrite | add species waves (passive, speed $u$) |
| Phase change, solid, embedded | 14 | rewrite | phase change is re-expressed on species (Stage 3) |

---

## 2. Model options

Notation: $N_p$ phases, phase $k$ has $N_k$ species, $s = \sum_k N_k$. $\eta_k$ volume fraction, $\rho_k$ phase density, $Y_{k,i}$ mass fraction of species $i$ within phase $k$ ($\sum_i Y_{k,i} = 1$), $m_{k,i} = \eta_k\rho_k Y_{k,i}$, $M_k \equiv (\eta\rho)_k = \sum_i m_{k,i}$.

### 2.1 Option (a): N phases, N volume fractions

Every chemical constituent is its own immiscible "phase" with its own volume fraction, density, pressure and internal energy. The six-equation model generalises directly (Saurel, Petitpas & Berry, J. Comput. Phys. 228 (2009) 1678; Schmidmayer, Bryngelson & Colonius, J. Comput. Phys. 402 (2020) 109080; implemented for N fluids in ECOGEN — Schmidmayer, Petitpas, Le Martelot & Daniel, Comput. Phys. Commun. 251 (2020) 107093 — and MFC — Bryngelson et al., Comput. Phys. Commun. 266 (2021) 107396):

$$\partial_t \eta_k + \mathbf{u}\cdot\nabla\eta_k = \mu\,(p_k - p_I),\qquad k = 1..N$$
$$\partial_t(\eta_k\rho_k) + \nabla\cdot(\eta_k\rho_k\mathbf{u}) = 0$$
$$\partial_t(\rho\mathbf{u}) + \nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u} + p\,\mathbf{I}) = 0,\qquad p = \sum_k \eta_k p_k$$
$$\partial_t E_k + \nabla\cdot(E_k\mathbf{u}) + \eta_k p_k\nabla\cdot\mathbf{u} = -\mu\,p_I\,(p_k - p_I)$$
$$\partial_t(\rho E) + \nabla\cdot((\rho E + p)\mathbf{u}) = 0$$

with $\sum_k\eta_k = 1$ (so $N-1$ independent volume fractions). State size $3N + D + 1$ (one of the $\eta_k$ redundant). The N-fluid form of the interfacial pressure $p_I$ and of the frozen sound speed $c^2 = \sum_k Y_k c_k^2$ follow the two-phase ones. **TODO verify** the exact $p_I$ ECOGEN uses for $N > 2$.

Properties:

- Each constituent keeps its own EOS, pressure and temperature; material interfaces between gases with different $\gamma$ are free of the pressure oscillations that conservative mass-fraction schemes produce (Abgrall, J. Comput. Phys. 125 (1996) 150).
- Constituents do **not** mix at the molecular level. Each occupies its own volume at its own temperature. A premixed H$_2$–O$_2$ gas in this model is a dispersion of H$_2$ volumes and O$_2$ volumes. Reaction rates need species concentrations in a common volume at a common temperature; that requires adding full p–T relaxation among all gas "phases" every step, after which the gas constituents are a Dalton mixture in all but bookkeeping — at the cost of $N$ energies and $N$ volume fractions instead of one each.
- Cost: 9-species H$_2$–O$_2$ plus a liquid gives $N = 10$, i.e. 30 + D + 1 evolved fields and a 10-wave-family HLLC star state, against 9 + 10 + D + 1 for option (b).
- Every one of the 1331 hard-coded lines changes.

### 2.2 Option (b): two phases, each a mixture of species

Keep the present six-equation set unchanged and add species partial densities inside each phase:

$$\partial_t \eta + \mathbf{u}\cdot\nabla\eta = \mu\,(p_1 - p_2)$$
$$\boxed{\ \partial_t m_{k,i} + \nabla\cdot(m_{k,i}\,\mathbf{u}) = -\nabla\cdot(\eta_k\,\mathbf{j}_{k,i}) + \eta_k\,\dot\omega_{k,i} + \Gamma_{k,i}\ }\qquad i = 1..N_k,\ k = 1,2$$
$$\partial_t(\rho\mathbf{u}) + \nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u} + p\,\mathbf{I}) = \nabla\cdot\boldsymbol\tau + \mathbf{S}_M$$
$$\partial_t E_k + \nabla\cdot(E_k\mathbf{u}) + \eta_k p_k\nabla\cdot\mathbf{u} = \mp\mu\,p_I\,(p_1 - p_2) + (\text{heat, diffusion, chemistry, phase-change terms})$$
$$\partial_t(\rho E) + \nabla\cdot((\rho E + p)\mathbf{u}) = \dots$$

- $\dot\omega_{k,i}$ [kg m$^{-3}$ s$^{-1}$ per unit volume *of phase k*]: net chemical production of species $i$ in phase $k$, $\sum_i\dot\omega_{k,i} = 0$.
- $\Gamma_{k,i}$: inter-phase transfer. For a substance $j$ present as liquid species $j_\ell$ and vapour species $j_v$: $\Gamma_{g,j_v} = -\Gamma_{\ell,j_\ell}$; zero for species with no counterpart.
- $\mathbf{j}_{k,i}$: diffusive mass flux within phase $k$, $\sum_i\mathbf{j}_{k,i} = 0$ (Stage 6; not needed for detonation).
- Summing over $i$ recovers the present phase-mass equations, with $\sum_i\Gamma_{k,i}$ the net phase-change mass source the `vapor` module already applies.

Phase EOS:

*Gas phase — ideal mixture of thermally perfect gases* (what PeleC does for the gas; Henry de Frahan et al., "PeleC: An adaptive mesh refinement solver for compressible reacting flows", Int. J. High Perform. Comput. Appl. 37 (2023) — **TODO verify** volume/pages):

$$p_g = \rho_g\,\frac{R_u}{\bar W}\,T_g,\qquad \frac{1}{\bar W} = \sum_i\frac{Y_i}{W_i},\qquad e_g = \sum_i Y_i\,e_i(T_g),\qquad e_i(T) = h_i(T) - \frac{R_u T}{W_i}$$

with $h_i(T)$ from NASA-7 polynomials (including the formation enthalpy), frozen sound speed $c_g^2 = \gamma_c\,p_g/\rho_g$, $\gamma_c = \bar c_p/\bar c_v$. $T_g$ is obtained from $(e_g, Y)$ by a scalar Newton.

*Liquid phase — ideal volume-additive mixture of stiffened-gas (or NASG) species* at common $p, T$, consistent with the `Phase` functions already in `Hydro2_PhaseChange.H:73-80`:

$$v_\ell = \sum_i Y_i\,v_i(p,T),\quad v_i = \frac{(\gamma_i-1)c_{v,i}T}{p+\pi_i}\ (+\,b_i\ \text{for NASG}),\qquad e_\ell = \sum_i Y_i\,e_i(p,T),\quad e_i = c_{v,i}T\,\frac{p+\gamma_i\pi_i}{p+\pi_i} + q_i$$

(stiffened gas with $q, q'$: Le Métayer, Massoni & Saurel, Int. J. Therm. Sci. 43 (2004) 265; NASG: Le Métayer & Saurel, Phys. Fluids 28 (2016) 046102). A single-species liquid reduces to the present Tammann phase.

Properties:

- This is the multi-component liquid–vapour–gas structure of Pelanti & Shyue (Int. J. Multiphase Flow 113 (2019) 208, **TODO verify** pages) and of Chiapolino, Boivin & Saurel (Comput. Fluids 150 (2017) 31, **TODO verify**) with the gas mixture made reactive.
- "One species may exist as a gas species and a liquid species treated separately" is native: `H2O` in the liquid block and `H2O` in the gas block are different components with different EOS, exchanged only through $\Gamma$.
- Species within a phase share that phase's pressure, temperature and volume, which is what a kinetics library expects ($\rho_g$, $T_g$, $Y_{g,i}$).
- The hyperbolic structure is unchanged: the species equations add $s$ linearly degenerate fields at speed $u$. HLLC, the `eta` row, the $E_k$ rows and the relaxation keep their form.
- Known weakness: inside a phase, a contact between species with different $\gamma$ in a conservative scheme generates pressure oscillations (Abgrall 1996; Karni, J. Comput. Phys. 112 (1994) 31). Reacting-flow codes accept this because diffuse gas–gas contacts make it small; a sharp gas–gas material interface (helium bubble in air) is better treated as two phases. See Section 7, test S5.

### 2.3 Option (c): hybrids

- (c1) **$N_p$ phases × species**: option (b) with $N_p > 2$ (e.g. liquid fuel, gas, solid/condensed product). Needed only if three immiscible media share cells. The elastic solid and the embedded solid in the present code are already handled by other mechanisms.
- (c2) **Option (b) plus a second gas phase** for cases with a sharp gas–gas interface; the same as (c1) with $N_p = 3$.
- (c3) **Five-equation (Allaire/Kapila) + species** (MFC-style reacting flow): gives up the six-equation robustness the dissertation is built on; not considered.

### 2.4 Recommendation

**Option (b), written so that the phase count is a loop bound (`NPHASE = 2`) in all new code**, which leaves (c1) open without committing to it.

Reasons, against the stated goals:

| Goal | Option (a) | Option (b) |
|---|---|---|
| Gaseous detonation (H$_2$–O$_2$–Ar and beyond) | gas constituents unmixed; needs p–T relaxation among ~9 "phases" each step to make kinetics meaningful | gas phase is exactly the thermally-perfect mixture detonation codes use; ZND/CJ references apply directly |
| Phase change of fuel droplets | one liquid↔vapour pair per pair of phases; non-condensables need a third phase | the existing `vapor` construction, generalised to M condensable species |
| Plasma later (more species, electrons, two temperatures) | each ion a phase with its own volume fraction — not physical for a plasma | more gas species; electrons a gas species; a second gas energy ($E_{ve}$ or $E_e$) is one more scalar in the gas phase |
| Cost at 9–20 species | $3N + D + 1$ fields, N-family star state | $s + D + 6$ fields, Riemann solver untouched |
| Refactor size | 1331 lines, all files | additive field + ~100 EOS call sites; Riemann/limiter files untouched |
| Bit-identity when off | hard: the data layout itself changes | straightforward: new field not registered, per-cell EOS cache not built |

What option (b) gives up — independent temperatures per species and oscillation-free gas–gas contacts — is recoverable case by case through (c1).

---

## 3. Data layout

### 3.1 What "`eta` as m × n × s" means under the recommendation

`eta` itself stays a one-component field (the liquid/gas volume fraction). The "× s" array is the **species partial-density field**:

```
species_mf      : Set::Field<Set::Scalar>, ncomp = s = N_0 + N_1, nghost = nghost, evolving = true
species_old_mf  : same, not written
component c(k,i) = offset[k] + i,   offset = {0, N_0}
value            = m_{k,i} = eta_k rho_k Y_{k,i}      [kg per m^3 of mixture]
```

Reasons for storing $m_{k,i}$ and not $Y_{k,i}$ or species volume fractions:

- It is the conserved quantity: flux-form update, `cell_cons_interp` on regrid and `average_down` are exactly conservative for it with no extra code, and the FluxRegister handles it like `rho_eta0/1`.
- It is well defined where the phase is absent ($m \to 0$ smoothly); $Y_{k,i}$ is 0/0 there.
- It is what `vapor` already is.

The species **volume fractions** the user described are a derived, plot-only field:

$$\eta_{k,i} = m_{k,i}\,v_{k,i}(p, T_k),\qquad \text{gas: } \eta_{g,i} = \eta_g X_i\ (\text{Amagat partial volume}),\qquad \sum_i\eta_{k,i} = \eta_k .$$

This is offered as `eta_s` (ncomp = s). It satisfies "the largest = 1 − the sum of the others" by construction once the closure of Section 4 is applied to the masses.

`rho_eta0_mf`/`rho_eta1_mf` are **kept** as the authoritative phase masses $M_k$. The species field is a decomposition of them, tied by $\sum_i m_{k,i} = M_k$ (Section 4). Keeping them means none of the 265 "mass" lines in `Hydro2.cpp` changes and `species.on = 0` leaves the state vector untouched.

### 3.2 Registration, names, restart

```cpp
if (value.species.on) {
    // suffix = {"_g_H2","_g_O2",...,"_l_H2O"}  -> plot names rhoY_g_H2, ...
    value.RegisterNewFab(value.species.m_mf,     value.species.bc, s, nghost, "rhoY",     true,  true, suffixes);
    value.RegisterNewFab(value.species.m_old_mf, value.species.bc, s, nghost, "rhoY_old", false, true);
    value.RegisterNewFab(value.species.diag_mf, &value.bc_nothing, n_diag, 0, "sp", true, false, diag_suffixes);
    // diag: Y_<name> (s), eta_s_<name> (s, optional), T_g, T_l, gamma_hat_k, pi_hat_k, and with chemistry: hrr, wdot_<name>
}
```

- Placed next to the `phasechange` block (`Hydro2.cpp` ≈895), and following its pattern: a helper class `Integrator::Species` in a new header `src/Integrator/Hydro2_Species.H` owning the fields, the name/offset tables, parse, and the per-cell kernels.
- Restart matches fields by name (`Integrator.cpp` ≈706 warns on names with no match). Species names in the suffix make a restart with a different species list fail visibly. **TODO verify** how multi-component fabs are named in the restart header before relying on this.
- Ghost cells: same width as the phase masses; filled in `FillGhost4BC` right after `rho_eta0/1` (≈7047). Recommended ghost fill: extrapolate/prescribe $Y_{k,i}$ and set $m_{k,i} = M_k^\text{ghost} Y_{k,i}$, so the ghost state satisfies the sum constraint whatever BC the phase masses use. A plain `BC::Expression` on $m$ would repeat the Dirichlet-sum problem documented at `Hydro2.H:411-418`.
- AMR: nothing special — `evolving = true`. After `average_down` the sum constraint still holds (linear). After `cell_cons_interp` it holds too if the interpolator's limiter is component-wise linear… which it is **not** in general (slopes are limited per component), so apply the Section 4 closure after `FillPatch` on new fine cells. `PostAverageDown` (≈8845) must recompute the EOS cache.
- Reflux: `ncomp_reflux` += s; new `cc_fluxes[lev].spec[d]` (ncomp s); one more `CrseInit`/`FineAdd`/`Reflux` triple (≈5019–5029, ≈8782).
- RK vector: append one entry `ix_sp` after `ix_pc` (≈4763), `RHS` gains `species_rhs_mf`/`species_mf_in` pointer arguments like the `vapor` pair (`Hydro2.H:160-161`).

### 3.3 Riemann `State` / `Flux` / `Primitive`

**No species members are added to `State`, `Flux` or `Primitive`.** Species are linearly degenerate at the contact; their flux is determined by the phase mass flux and an upwind mass fraction:

$$F_{k,i} = F_{M_k}\,\hat Y_{k,i}^{\,\text{up}},\qquad F_{M_k} = \texttt{flux.mass}k .$$

This is computed in `RHS` after the Riemann solve, where `vapor` is computed now. All 11 Riemann solver files stay as they are.

Species mass fractions are reconstructed with the selected limiter through the existing passive path, `Limiter::ReconstructPassive` routed through the `alpha` slot (`Limiter.H:232-235`; this is how the shell $\Gamma$ is reconstructed). For the in-kernel stencil a fixed-size array is used:

```cpp
#ifndef HYDRO2_NSPEC_MAX
#define HYDRO2_NSPEC_MAX 16          // compile-time cap; make variable
#endif
struct SpeciesFace { Set::Scalar Y[HYDRO2_NSPEC_MAX]; };   // trivially copyable, no heap
```

Fixed-size vs dynamic:

- A `std::vector` inside a `ParallelFor` body allocates per cell and cannot go to a GPU. `std::array`/C array with a compile-time cap is the AMReX-standard choice (PelePhysics does the same with `NUM_SPECIES` from the generated `mechanism.H`; the Makefile comment at `Makefile:68-71` already notes that the mechanism owns that define).
- With PelePhysics enabled, `HYDRO2_NSPEC_MAX` must be ≥ `NUM_SPECIES` of the chosen mechanism + the number of liquid species; a `static_assert` enforces it. The runtime count `s ≤ HYDRO2_NSPEC_MAX` comes from the deck.
- GPU: Hydro2 is not GPU-ready today for independent reasons (virtual `limiter->Reconstruct` / `riemannsolver->Solve` inside kernels; host-static EOS backend, `EOS.H:30-35`; `configure` has no CUDA/HIP option that I found). The species code should not add new obstacles: POD structs, `AMREX_GPU_HOST_DEVICE` free functions, no virtual calls, no host statics. Mutation++ (Section 6) is host-only by construction.

### 3.4 How a species-dependent EOS reaches the existing Tammann algebra

The hyperbolic step, the limiter back-conversion and the relaxation all assume, per phase,

$$p_k = (\gamma_k-1)\,\frac{E_k}{\eta_k} - \gamma_k\pi_k,\qquad c_k^2 = \frac{\gamma_k(p_k+\pi_k)}{\rho_k}.$$

Proposal (mine — in the spirit of the effective-$\gamma$ treatment of real gases by Colella & Glaz, J. Comput. Phys. 59 (1985) 264; **TODO verify** whether an existing paper uses exactly this two-parameter fit): at the start of each `RHS` evaluation compute, per cell and per phase, the **local stiffened-gas fit** that reproduces the true phase pressure, sensible internal energy and frozen sound speed at that cell's state. With $A = \rho_k c_k^2$ and $B = \rho_k e_k^s$ (sensible energy per unit phase volume, Section 6.5):

$$\hat\gamma_k = \frac{p + A + B}{p + B},\qquad \hat\pi_k = \frac{A}{\hat\gamma_k} - p .$$

Checks: (i) for a pure stiffened gas, $A = \gamma(p+\pi)$, $B = (p+\gamma\pi)/(\gamma-1)$ give $\hat\gamma = \gamma$, $\hat\pi = \pi$ exactly; (ii) for a calorically perfect gas $\hat\gamma = \gamma$, $\hat\pi = 0$; (iii) for a thermally perfect gas with $\gamma_e \equiv 1 + p/(\rho e^s) \ne \gamma_c$, $\hat\gamma = ((\gamma_e-1)(1+\gamma_c)+1)/\gamma_e$ and $\hat\pi = p(\gamma_c/\hat\gamma - 1)$, a small fraction of $p$ of either sign with $p + \hat\pi = \gamma_c p/\hat\gamma > 0$.

Wiring:

- A per-level scratch MultiFab `eoshat_mf` (4 components: $\hat\gamma_0, \hat\pi_0, \hat\gamma_1, \hat\pi_1$; `nghost` ghosts) filled after the ghost fill.
- `get_prim` (≈2539–2540) reads `q.gamma0 = eoshat(i,j,k,0)` … instead of `eos0.Gamma()`; the primitive cache (≈2477–2512) uses the same values. `Limiter::ToState` rebuilds $E_k$ at a face with the *centre cell's* $(\hat\gamma,\hat\pi)$ (metadata is taken from the LHS, `Limiter.H:81-82`), which is first-order consistent in the EOS variation and exact for single-species phases.
- `RelaxAndReinit`, `PostAverageDown`, `FillGhost4BC`, the $E_k$ rows and NSCBC read the same cache instead of `eosK.Gamma()/P0()` — these are the ~100 EOS-constant lines of Section 1.9.
- With `species.on = 0` the cache is not allocated and the code reads `eos0`/`eos1` as now (bit-identical). With `species.on = 1` and one constant-$\gamma$ species per phase the fit returns the deck constants up to round-off (a consistency test, not bit-identity).

What stays approximate: the fit is frozen across each Riemann fan and during each relaxation solve. The star-state phase energies are therefore approximate, as they already are (they are overwritten by the reinit from the conserved $\rho E$, which uses the true EOS — Section 5.1).

### 3.5 Cost estimate

Formulae, not measurements:

- Evolved primaries today: $8 + D$ components (10 in 2D). With species: $8 + D + s$. For $s = 10$ in 2D the RK-carried state doubles. Hydro2 registers on the order of 60 other fields (Section 1.1) that do not grow, so total memory grows by much less than 2×; measure with a real deck.
- Hyperbolic cost: per face and species, two passive reconstructions and one multiply, plus the per-cell EOS cache (one Newton for $T_g$, 3–5 polynomial evaluations × $N_g$). Expect the non-chemistry step to grow roughly linearly in $s$ with a small coefficient.
- Chemistry: one stiff ODE solve per reacting cell per step, $N_g + 1$ unknowns, dense Jacobian factorisation $O(N_g^3)$ per Newton matrix. In reacting-flow codes this is normally the dominant cost once the mechanism has more than ~10 species; gating on $\eta_g$, $T$ and fuel presence is essential.

---

## 4. Advection and bounds

### 4.1 The user's rule, stated precisely

> Advect the smallest fractions first; let the largest be 1 − the sum of the others, chosen per cell.

Two places need a closure, and the rule is applied at both.

**(A) Face mass fractions (flux consistency).** For each face, each phase $k$, each side (L, R):

1. $Y_{k,i}^c = m_{k,i}/M_k$ at the cell centres of the stencil (set to the neighbour's value, or 0, where $M_k$ is below a floor).
2. Let $i^\star = \arg\max_i Y_{k,i}^{c}$ **of that side's own cell** (L uses the lo cell, R uses the hi cell).
3. For every $i \ne i^\star$: $\hat Y_{k,i} = $ limiter reconstruction (`ReconstructPassive`), clipped to $[0,1]$.
4. $\hat Y_{k,i^\star} = 1 - \sum_{i\ne i^\star}\hat Y_{k,i}$, summing in ascending order of $Y^c$ (smallest first — this is where the ordering has a numerical effect: it minimises round-off in the sum).
5. If $\hat Y_{k,i^\star} < 0$ (possible only if the limited minor species overshoot collectively), fall back for that side to proportional rescaling $\hat Y_i \leftarrow \hat Y_i/\sum_j\hat Y_j$ and count the event.

Then, with the phase mass flux from HLLC,

$$F_{k,i} = F_{M_k}\times\begin{cases}\hat Y_{k,i}^{L}, & F_{M_k} > 0\\ \hat Y_{k,i}^{R}, & F_{M_k} \le 0\end{cases}\qquad\Longrightarrow\qquad \sum_i F_{k,i} = F_{M_k}\ \text{exactly (to round-off)}.$$

**(B) Cell closure after each stage** (in the post-stage hook, before relaxation):

1. Clip $m_{k,i} < 0$ to 0 for $i \ne i^\star_\text{cell}$ (count and accumulate the clipped mass as a diagnostic).
2. $m_{k,i^\star} = M_k - \sum_{i\ne i^\star}m_{k,i}$ with $i^\star$ the cell's largest species.

### 4.2 Properties

*Conservation of each species.* The update is in flux form with a single-valued face flux, so $\sum_\text{cells} m_{k,i}$ changes only by boundary fluxes and sources, for every species, whichever closure is used at the face. The face rule changes accuracy and positivity, not conservation. The cell closure (B) alters only the majority species, by the drift $M_k - \sum_i m_{k,i}$. That drift is zero to round-off when fluxes satisfy (A) and sources sum to zero, and is otherwise exactly the clipped mass. Trace species are never rescaled.

*Consistency with total mass.* Summing the species update over $i$ reproduces the `rho_eta_k` update exactly because $\sum_i F_{k,i} = F_{M_k}$. So $\sum_i Y_{k,i} = 1$ is preserved to round-off without renormalisation. This is the property the "consistent multi-fluid advection" (CMA) of Plewa & Müller (Astron. Astrophys. 342 (1999) 179) enforces; they do it by rescaling all face fractions by $1/\sum_j\hat Y_j$.

*Why majority-closure and not CMA rescaling.* CMA multiplies every species' face value — including trace species at $10^{-10}$ — by a factor $1/\sum\hat Y$ that differs from 1 by the reconstruction mismatch (percent-level at steep fronts). That is still conservative, but it perturbs the trace fluxes by a relative amount set by the *major* species' reconstruction error, and can break the monotonicity each limited $\hat Y_i$ had on its own. Under the majority closure each minor species keeps its own limited (monotone) reconstruction untouched, and the whole mismatch goes to the species that can best absorb it. This is the content of the user's argument and I agree with it. CMA rescaling is kept as the fallback of step A5.

*Positivity.* With first-order upwind (Godunov) fluxes and the hydrodynamic CFL, each $m_{k,i} \ge 0$ is preserved provided $M_k \ge 0$ is (the species update is a convex combination with the same weights as the phase mass update). With a higher-order limiter, positivity holds for the minor species as far as the limiter is monotone in $Y$; the majority is positive as long as $\sum_{i\ne i^\star}\hat Y_i \le 1$, guarded by A5. The cell clip B1 catches what remains; its count must stay at zero in the verification tests (Section 7, S1–S3).

*What "largest" means when it changes across a face.* Nothing needs to agree across the face. The L state is a statement about the lo cell's sub-cell profile and uses the lo cell's majority; the R state uses the hi cell's. The upwind choice then picks one complete, sum-consistent set. The flux is single-valued, so conservation is unaffected. Where the majority switches from cell to cell (e.g. across a fuel–oxidiser contact) both candidate majority species are large, so either is well away from zero.

*Does the order "smallest first" matter beyond the sum?* In an unsplit explicit flux-form update the species fluxes are independent; only "which species is the closure" matters, and that needs an `argmax`, not a sort. A full ascending order matters in two places: the round-off of the closure sum (A4), and a sequential positivity limiter that hands out a cell's available outflow budget species by species (not proposed for the first implementation). The implementation should therefore do an `argmax` plus an ascending-order sum; a full sort per face is unnecessary cost for $N \le 16$ (an insertion sort is fine if wanted).

*Interface cells.* $Y_{k,i} = m_{k,i}/M_k$ is ill-conditioned where phase $k$ is a trace. Use the same guard the `vapor` code uses (`mg > 1e-300`, ≈4091) but with a physical floor (`M_k < species.mass_floor` → take $Y$ from the nearest cell where the phase is present, as the shell bulk-extension does). The `RHO_PURE_CAP` discussion in `Limiter.H:130-162` applies: bound the result of the divide, not the divisor.

*The `eta` row.* Unchanged. If the code is later extended to $N_p > 2$ phases (option c1), the same rule applies to the volume fractions: evolve all $\eta_k$, set the per-cell majority $\eta_{k^\star} = 1 - \sum_{k\ne k^\star}\eta_k$. There the equation is non-conservative, so a closure is required rather than optional. MFC has an option of this kind for the last volume fraction — **TODO verify** name and behaviour.

### 4.3 What other codes do (from memory unless stated)

- **PeleC**: species partial densities $\rho Y_i$ are conserved variables; Godunov states carry mass fractions; species flux = mass flux × upwind face mass fraction. Negative/unnormalised mass fractions are repaired by clipping and renormalising. **TODO verify** function names (`pc_cmpflx`, `clean_massfrac`) against the source once cloned.
- **CASTRO**: same passive-upwind species flux; an optional step normalises the species fluxes to sum to the density flux, citing Plewa & Müller (1999); mass fractions are clipped to `[small_x, 1]` and renormalised (`normalize_species`). **TODO verify** names.
- **Plewa & Müller (1999)**: the CMA reference; also discusses why renormalising cell values after the update (instead of fluxes) is not conservative.
- **Reacting DNS codes** commonly solve $N-1$ species with a fixed inert (N$_2$) as $1 - \sum$. That is the user's rule with a globally fixed majority; a per-cell majority removes the need for an inert that is abundant everywhere, at the price of storing all $N$.

---

## 5. Relaxation with N components

Hierarchy and order within a step, stiffest first (this is the order of Pelanti & Shyue, J. Comput. Phys. 259 (2014) 331, and of Saurel, Petitpas & Abgrall, J. Fluid Mech. 607 (2008) 313): hyperbolic step → **mechanical** ($p$) → **thermal** ($p$–$T$) → **chemical potential** ($p$–$T$–$\mu$, phase change) → **finite-rate chemistry** (ODE). Each later stage starts from a state that already satisfies the earlier equilibria, which is what keeps the Newton iterations short.

Fixed during every cell solve: the total species masses, the cell volume, momentum, and total energy $\rho E$.

### 5.1 Mechanical relaxation — still scalar

For any number of phases $N_p$ and any number of species per phase, the pressure-relaxation unknown is the single equilibrium pressure:

$$f(p) = \sum_{k=1}^{N_p} M_k\,v_k(p) - 1 = 0,$$

with $v_k(p)$ the closed form of Section 1.5 evaluated with the local fit $(\hat\gamma_k,\hat\pi_k)$ of Section 3.4. Species composition is frozen during the solve and enters only through the fit. $dv_k/dp < 0$ for every phase with $p_k^0 + \hat\pi_k > 0$, so $f$ is strictly decreasing on $(-\min_k\hat\pi_k, \infty)$ and the root is unique: **the existing Newton + bisection carries over with the two hand-written terms replaced by a loop.** No Jacobian matrix is involved. (ECOGEN's N-phase pressure relaxation is, as far as I recall, this same single-unknown procedure — **TODO verify** against the ECOGEN source/paper.)

Changes needed:

- Loop over `k < NPHASE` instead of `0/1` temporaries.
- $p_\text{min} = -\min_k\hat\pi_k$ with a per-cell $\hat\pi$ that can be slightly negative for a thermally perfect gas (Section 3.4 iii) — the floor expressions `-std::min(pi0_, pi1_)` (≈8395, 8576) must use the cell values.
- **Reinit with the true EOS.** After $\eta_k^\text{new}$ is known, find $p$ such that the phase energies at $(\rho_k = M_k/\eta_k^\text{new},\ p,\ Y_k)$ sum to the conserved sensible energy:
$$g(p) = \sum_k \eta_k^\text{new}\,\rho_k\,e_k^s(\rho_k, p, Y_k) - \rho e^s = 0,\qquad \frac{dg}{dp} = \sum_k\frac{\eta_k^\text{new}}{\mathcal{G}_k},\quad \mathcal{G}_k \equiv \frac{1}{\rho_k}\Big(\frac{\partial p}{\partial e}\Big)_{\rho_k,Y}$$
($\mathcal{G}_k = \gamma_k - 1$ for a stiffened gas; $R_u/(\bar W\bar c_v(T))$ for the ideal mixture). For single-species stiffened-gas phases $g$ is linear in $p$ and the solve is the present `ReinitMixturePressure`. For a thermally perfect gas it is a scalar Newton that converges in 2–3 iterations from the relaxed $p^*$; $dg/dp > 0$ always.
- Trace handling: the pure-cell guard (≈8322) is unchanged. Species do not add new trace cases to the mechanical step because composition is frozen in it.

### 5.2 Thermal relaxation ($p$–$T$) — 2×2 Newton

Needed as the inner state of phase change, and optionally on its own (`species.thermal_relax`) in interface cells before chemistry. Unknowns $\mathbf{x} = (p, T)$. Known: all $m_{k,i}$ and $\rho e$ (absolute, i.e. including $q_i$/formation energies — Section 6.5).

$$R_1 = \sum_k\sum_i m_{k,i}\,v_{k,i}(p,T) - 1,\qquad R_2 = \sum_k\sum_i m_{k,i}\,e_{k,i}(p,T) - \rho e .$$

Jacobian $J = \begin{pmatrix}\partial_p R_1 & \partial_T R_1\\ \partial_p R_2 & \partial_T R_2\end{pmatrix}$ assembled from per-species derivatives:

| Species type | $v_i$ | $\partial_p v_i$ | $\partial_T v_i$ | $e_i$ | $\partial_p e_i$ | $\partial_T e_i$ |
|---|---|---|---|---|---|---|
| ideal gas (in mixture at total $p$) | $R_uT/(W_ip)$ | $-v_i/p$ | $v_i/T$ | $e_i(T)$ (polynomial) | 0 | $c_{v,i}(T)$ |
| stiffened gas | $(\gamma_i-1)c_{v,i}T/(p+\pi_i)$ | $-v_i/(p+\pi_i)$ | $v_i/T$ | $c_{v,i}T\frac{p+\gamma_i\pi_i}{p+\pi_i}+q_i$ | $-\dfrac{c_{v,i}T(\gamma_i-1)\pi_i}{(p+\pi_i)^2}$ | $c_{v,i}\dfrac{p+\gamma_i\pi_i}{p+\pi_i}$ |
| NASG | SG form $+\,b_i$ | as SG | as SG | as SG | as SG | as SG |

(For the ideal-gas rows, $\sum_i m_{g,i}v_i = \eta_g$ is Amagat's law: each species' partial volume at the total pressure.)

When every species has constant $c_v$, $R_2$ is linear in $T$ and $T(p)$ can be eliminated — that is the present `ptEquilibrium` (`Hydro2_PhaseChange.H:108-139`), a scalar solve. With NASA polynomials $e_i(T)$ is a quartic in $T$ and the 2×2 Newton is the natural form.

### 5.3 Chemical-potential relaxation ($p$–$T$–$\mu$) with M condensable species — (2+M)×(2+M) Newton

For each substance $j = 1..M$ present in both phases, let $\xi_j$ be its vapour partial density; its liquid partial density is $m_j^\text{tot} - \xi_j$. Species without a counterpart (non-condensable gases, non-volatile liquids) are fixed. Unknowns $\mathbf{x} = (p, T, \xi_1..\xi_M)$.

$$R_1 = \sum_{k,i}m_{k,i}(\boldsymbol\xi)\,v_{k,i}(p,T) - 1,\qquad R_2 = \sum_{k,i}m_{k,i}(\boldsymbol\xi)\,e_{k,i}(p,T) - \rho e,$$
$$R_{2+j} = \mu_{\ell,j}(p,T,\mathbf{x}_\ell) - \mu_{g,j}(p,T,\mathbf{x}_g),\qquad j = 1..M,$$

with, for ideal mixtures (mole fractions $x$),

$$\mu_{g,j} = g_{j}^{v}(x_{g,j}\,p,\ T),\qquad \mu_{\ell,j} = g_{j}^{\ell}(p, T) + \frac{R_u T}{W_j}\ln x_{\ell,j},$$

where $g^{v}_j, g^{\ell}_j$ are the pure-species Gibbs functions (`PhaseChange::gibbs`, `Hydro2_PhaseChange.H:79-80`, for stiffened-gas species; from the thermo library for NASA-polynomial species). For $M = 1$ and a pure liquid this is the present condition $g_\ell(p,T) = g_v(x_v p, T)$ (`Hydro2_PhaseChange.H:39`, `:158`).

Jacobian structure:

$$J = \begin{pmatrix}
\partial_p R_1 & \partial_T R_1 & v_{g,j} - v_{\ell,j}\ \ (j=1..M)\\
\partial_p R_2 & \partial_T R_2 & e_{g,j} - e_{\ell,j}\\
\partial_p\Delta\mu_j & \partial_T\Delta\mu_j & \partial_{\xi_{j'}}\Delta\mu_j
\end{pmatrix}$$

- The first two rows' $\xi_j$ columns are the specific volume and energy differences of substance $j$ between its two forms (the latent heat appears in $e_{g,j} - e_{\ell,j}$ through $q$).
- $\partial_p g = v$, $\partial_T g = -s$ for each pure-species Gibbs function, so $\partial_p\Delta\mu_j = v_{\ell,j} - v_{g,j}^{(x_j p)}x_{g,j}$ etc.; all analytic.
- The $M\times M$ block is dense: changing $\xi_{j'}$ changes every mole fraction in both phases. $\partial x_{g,j}/\partial\xi_{j'} = (\delta_{jj'} - x_{g,j})/(W_{j'}n_g)$ with $n_g = \sum_i m_{g,i}/W_i$, similarly (opposite sign) in the liquid. This block is where "how every species interacts with every other" lives.

Solver (new header `src/Solver/Local/Relax/Newton.H`, fixed-size, kernel-safe):

1. **Scaling.** Solve in $(\ln(p + \pi_\text{ref}),\ \ln T,\ \xi_j/m_j^\text{tot})$ and scale residuals by (1, $\rho e_\text{ref}$, $R_uT/W_j$) so the merit function is dimensionless. Stiffened liquids put $10^9$ Pa next to $10^{-9}$ mass fractions; unscaled Newton stalls, as the comment at `Hydro2.cpp` ≈8402 already records for the scalar case.
2. **Newton step** with dense LU on a fixed-size $(2+M)$ system (hand-written partial-pivot LU; Eigen fixed-size is an alternative on CPU).
3. **Backtracking line search** on $\phi = \tfrac12\|\mathbf{R}_\text{scaled}\|^2$ (Armijo), with the step first truncated so that $p > -\min\pi$, $T \in [T_\text{lo}, T_\text{hi}]$, $0 \le \xi_j \le m_j^\text{tot}$.
4. **Active set for bounds.** If a truncated step leaves $\xi_j$ at a bound and $\Delta\mu_j$ has the sign that pushes it further out (liquid fully evaporated, or no vapour and undersaturated), drop equation $2+j$ and fix $\xi_j$ (the present code's "or m_vap = 0 / m_liq = trace at a bound"). Re-admit when the sign reverses.
5. **Fallback.** If Newton fails to reduce $\phi$ in `max_iter` iterations: nonlinear Gauss–Seidel over the substances, each sweep using the existing bracketed scalar solver (`PhaseChange::relax`) for one $\xi_j$ with the others frozen, ordered by decreasing $|\Delta\mu_j|/(R_uT/W_j)$ — the furthest-from-equilibrium (stiffest) first. Each scalar solve is bracketed and cannot diverge. If that also fails, leave the cell unchanged and count it (the present behaviour, `return false`).
6. **Trace species.** A species with $m < $ `species.trace` × $\rho$ in *both* phases is removed from the unknowns (its $\ln x$ is undefined and it cannot influence $p, T$). A phase with $\eta_k < $ `alpha_min` is skipped as now (`Hydro2.cpp` ≈8713).

What the literature does (**all TODO verify** against the papers before citing in the dissertation): Pelanti & Shyue (2014; 2019) reduce $p$–$T$ and $p$–$T$–$g$ for stiffened-gas phases to single-variable equations and use Newton on those; Le Métayer, Massoni & Saurel (ESAIM: Proc. 40 (2013) 103) formulate the liquid–vapour–non-condensable equilibrium; Chiapolino, Boivin & Saurel (2017) replace the iterative solve by an algebraic approximate equilibrium for speed. ECOGEN offers P, PT and PTMu relaxation, the last for one liquid/vapour pair. I am not aware of a published diffuse-interface code doing the general $M > 1$ coupled Newton above; if that is confirmed by a literature check it is a candidate contribution in itself.

### 5.4 How the present two-phase code generalises

| Present | Generalisation |
|---|---|
| `RelaxAndReinit`: scalar Newton + bisection on $f(p)$, two hand-written phases | loop over phases; per-cell $(\hat\gamma,\hat\pi)$; reinit by scalar Newton on $g(p)$ with the true EOS |
| `PhaseChange::ptEquilibrium`: scalar in $p$, $T(p)$ eliminated | §5.2 2×2 Newton; reduces to the present solve for constant-$c_v$ species (kept as the fallback) |
| `PhaseChange::relax`: 1 unknown ($m_\text{vap}$), nested bracketing | §5.3 with $M = 1$; present solver kept as the Gauss–Seidel fallback and as the regression reference |
| `vapor` field | species `H2O` in the gas block; `phasechange.Yv0` → gas species IC |
| equal molar masses assumed (`:39`) | mole fractions from $W_i$ |

---

## 6. Chemistry coupling

### 6.1 What is on this machine

| Item | Found | Where |
|---|---|---|
| PelePhysics source | **no** | `ext/` contains only `AMReX-Codes/` and `libeigen/` |
| SUNDIALS | **no** | `ext/sundials` absent; `find / -maxdepth 6 -iname "*sundials*"` empty |
| Mutation++ | **no** | `find / -maxdepth 6 -iname "*mutation*"` returns only an unrelated Ruby file |
| Cantera (reference solutions) | **no** | `python3 -c "import cantera"` → `ModuleNotFoundError` |
| SDToolbox / NASA CEA | **no** | not found |
| Eigen | yes | `ext/libeigen/eigen3` |
| PelePhysics **scaffold** in this repo | yes | `Makefile:53-85` (`ifdef PELE_ENABLED`), `scripts/build_pele_deps.sh`, `src/Solver/EOS/PelePhysicsEOS.H`, `src/pele_hello.cc`, `EOS.H:12-14, 21-76, 460-509`, `eos.backend` parse at `Hydro2.cpp:449` |
| `bin/PeleC.md` (the roadmap those files point to) | **missing** | referenced at `Makefile:65`, `EOS.H:34`, `PelePhysicsEOS.H:37`; not on disk in `flames`, `flames2` or `flames_wedge_baseline` |

Network: outbound HTTPS works from this session (GitHub and readthedocs pages were fetched). Obtaining PelePhysics, SUNDIALS, Mutation++, Cantera or SDToolbox requires a download; **nothing was downloaded**. `scripts/build_pele_deps.sh` expects `ext/sundials` (it suggests `v7.1.1`) and `ext/pelephysics` to be cloned by hand first and then builds SUNDIALS with CMake.

Git history shows the scaffold was once exercised (`0f8e746ed "PeleC added, passes Garrick and Toro1a"`), with the single-species `GammaLaw` EOS only.

### 6.2 PelePhysics vs Mutation++

V = verified during this investigation from the project's own pages; M = from memory, **TODO verify** against the source once it is on disk.

| | PelePhysics | Mutation++ |
|---|---|---|
| Origin | AMReX-Combustion (NREL/LBNL/Sandia/ORNL) | von Karman Institute (V) |
| Licence | BSD-3-Clause (V, `license.txt`) | LGPL v3 (V, README) |
| Language / build | C++, GNUmake integrated with AMReX's make system; no standalone library — sources compile into the application (V: stated in this repo's `build_pele_deps.sh:84-87`); CMake also exists (M) | C++, CMake, builds `libmutation++` (V: CMake; M: library name); needs Eigen (M) |
| Dependencies | AMReX and SUNDIALS (V) | Eigen; data directory via `MPP_DATA_DIRECTORY` (M) |
| Thermodynamics | generated code from a Cantera-YAML mechanism via CEPTR (V); NASA-7 fits (M); EOS: GammaLaw, ideal-gas mixture ("Fuego"), Soave–Redlich–Kwong (V), Manifold (seen in repo comment `PelePhysicsEOS.H:63-67`) | NASA-7, NASA-9 and rigid-rotor/harmonic-oscillator databases (M) |
| Kinetics | generated production-rate routines; **analytic Jacobian** generated by CEPTR (V) | finite-rate production rates (V); Jacobian w.r.t. species densities (M: `jacobianRho`) |
| ODE integrator | CVODE (V), ARKODE (V), explicit RK64 and a null reactor (M); constant-volume ($\rho$, $e$) and constant-volume-enthalpy reactors (V); dense direct, sparse direct (KLU), GMRES linear solvers (V) | **none** |
| Transport | constant, simple mixture-averaged (EGLib-consistent), Soret (V) | Chapman–Enskog multicomponent transport including ionised mixtures (M; "transport properties" V) |
| Plasma / ionisation / two-temperature | not a design target; no two-temperature model (M). Mechanisms with charged species exist for the PeleLMeX electric-field module (M) | **yes** — the library's purpose is ionised gases: electrons and ions as species, state models incl. `ChemNonEq1T`, `ChemNonEqTTv`, energy-transfer source terms, multiphase equilibrium solver, gas–surface interaction (V for the capability list; M for model names) |
| GPU | yes, via AMReX (CUDA/HIP/SYCL) with CVODE on device (docs: "technically available", V) | no; object-oriented with virtual dispatch and a stateful `Mixture` (M). One `Mixture` object per OpenMP thread is required (M) |
| Units | CGS in the EOS/kinetics routines (M — important; Hydro2 is SI) | SI (M) |
| Species count | compile-time `NUM_SPECIES` from the generated `mechanism.H` (V: `Makefile:68-71`) | run-time from the mixture file |
| Shipped H$_2$ mechanism | yes (README lists hydrogen; M: `LiDryer`, 9 species incl. N$_2$) | air/CO$_2$/planetary-entry mixtures (M); combustion mechanisms would be user-supplied |

How each is driven (M — signatures to be checked against headers):

```cpp
// PelePhysics (per cell, inside a kernel; CGS)
auto eos = pele::physics::PhysicsType::eos();
eos.REY2T(rho, e, Y, T);            // T from (rho, e, Y)
eos.RTY2P(rho, T, Y, p);            // pressure
eos.RTY2Cs(rho, T, Y, cs);          // frozen sound speed
eos.TY2Cv(T, Y, cv);  eos.T2Ei(T, ei);  eos.T2Hi(T, hi);
eos.RTY2WDOT(rho, T, Y, wdot);      // species production rates
// reactor: pele::physics::reactions::ReactorBase::create("ReactorCvode");
//          reactor->init(reactor_type, ncells);
//          reactor->react(box, rhoY, frcExt, T, rhoE, frcEExt, fc, mask, dt, time);

// Mutation++ (host only; SI)
Mutation::MixtureOptions opts("air_11");  opts.setStateModel("ChemNonEq1T");
Mutation::Mixture mix(opts);
mix.setState(rho_i, &rho_e, 0);     // species densities + energy density  (or (rho_i, &T, 1))
mix.T(); mix.P(); mix.frozenSoundSpeed(); mix.mixtureFrozenCvMass();
mix.netProductionRates(wdot);       // kg/m^3/s
mix.speciesHOverRT(h); mix.viscosity(); mix.frozenThermalConductivity(); mix.averageDiffusionCoeffs(D);
mix.energyTransferSource(omega);    // for two-temperature models
```

### 6.3 Linking into this build

The top-level `Makefile` compiles every `src/**/*.cpp` (`SRC = $(shell find src/ -mindepth 2 -name "*.cpp")`) and links `libamrex.a` from `AMREX_TARGET`; AMReX is consumed as an installed library, not through AMReX's GNUmake. `configure` writes compiler flags to a generated include file. Every `src/*.cc` becomes its own executable (that is how `pele_hello` is built), which gives a place for stand-alone unit-test programs.

*PelePhysics.* The existing block (`Makefile:67-85`) adds include paths for `Source/`, `Source/Eos`, `Utility/BlackBoxFunction`, `Mechanisms/air`, and links CVODE. To go beyond `GammaLaw`:

1. Make the mechanism selectable: `PELE_MECHANISM ?= LiDryer`; `-isystem ${PELE_PHYSICS_HOME}/Mechanisms/$(PELE_MECHANISM)`; replace `-DUSE_GAMMALAW_EOS` by `-DUSE_FUEGO_EOS`.
2. Compile the mechanism's generated `mechanism.cpp` into the object list (an explicit extra `OBJ` under `ifdef PELE_ENABLED`; PelePhysics provides no library to link).
3. Decide how much of PelePhysics to use:
   - **(recommended) thermo + kinetics only**: `mechanism.H/.cpp` and the EOS headers, with Hydro2 driving CVODE itself (one small dense system per cell, serial `N_Vector`, one CVODE memory block per OpenMP thread). This avoids PelePhysics' reactor layer, which is written against AMReX's own SUNDIALS integration (`amrex::sundials` context) and so, as far as I recall, needs **AMReX itself rebuilt with SUNDIALS enabled** — **TODO verify**. It also avoids the header-name collision already documented in `PelePhysicsEOS.H:45-58` (PelePhysics' `EOS.H` vs `src/Solver/EOS/EOS.H`) if the include is done from a translation unit that does not include Hydro2's `EOS.H` by a bare name.
   - full PelePhysics reactors (`ReactorCvode`): more capable (GPU, sparse solvers) but couples the AMReX build to SUNDIALS.
4. The AMReX-version incompatibility noted at `PelePhysicsEOS.H:60-68` (`EosParams.H` vs AMReX 25.04) means the PelePhysics commit must be **pinned** to one compatible with `ext/AMReX-Codes/amrex`.
5. Unit conversion SI↔CGS at the shim boundary ($\rho$: ×10⁻³, $e$: ×10⁴, $p$: ×10, $\dot\omega$: ×10³ back), in one place.

*Mutation++.* Build with CMake into `ext/mutationpp/install` (a new `scripts/build_mpp_deps.sh`), then

```make
ifdef MPP_ENABLED
    CXX_COMPILE_FLAGS += -DMPP_ENABLED
    ALAMO_INCLUDE += -isystem ${MPP_HOME}/install/include
    LIB += -L${MPP_HOME}/install/lib -lmutation++
endif
```

plus `MPP_DATA_DIRECTORY` at run time. LGPL permits dynamic linking from this code without licence consequences for it; static linking has conditions — a question for the user if the binary is ever distributed.

### 6.4 The interface Hydro2 should code against

One abstraction, three backends, so Hydro2 never calls a library directly:

```cpp
namespace Solver { namespace Chem {
struct Thermo {            // per gas-phase cell; all SI; fixed-size arrays
    int  n;                                    // number of gas species
    // species data
    Set::Scalar W(int i) const;                // molar mass [kg/kmol]
    void  ei (Set::Scalar T, Set::Scalar *e)  const;   // absolute specific internal energies [J/kg]
    void  cvi(Set::Scalar T, Set::Scalar *cv) const;
    void  gi (Set::Scalar T, Set::Scalar p, Set::Scalar *g) const;   // Gibbs, for phase change
    // mixture
    Set::Scalar T_from_e(Set::Scalar e, const Set::Scalar *Y, Set::Scalar Tguess) const;
    // kinetics
    void  wdot(Set::Scalar rho, Set::Scalar T, const Set::Scalar *Y, Set::Scalar *w) const;      // [kg/m^3/s]
    void  jac (Set::Scalar rho, Set::Scalar T, const Set::Scalar *Y, Set::Scalar *J) const;      // d(wdot,Tdot)/d(rhoY,T), optional
};
}}
```

| Backend | Purpose | Needs external code |
|---|---|---|
| `native` | constant-$c_p$ species ($\gamma_i, W_i, h_{f,i}$) + user-listed Arrhenius reactions (enough for one-step $A\to B$) | no — gives the closed-form tests of Section 7 |
| `pelephysics` | NASA-7 thermo, real mechanisms, analytic Jacobian | PelePhysics + SUNDIALS |
| `mutationpp` | ionised mixtures, two-temperature, equilibrium composition | Mutation++ |

State passed from Hydro2 to the chemistry, per cell with $\eta_g >$ `chem.alpha_min`:

$$\rho_g = M_g/\eta_g,\qquad Y_{g,i} = m_{g,i}/M_g,\qquad e_g = E_g^\text{abs}/M_g\ \ (\text{or } T_g).$$

Returned after integrating over $\Delta t$ at constant gas volume: new $Y_{g,i}$ (hence $m_{g,i}$) and $T_g$; the gas absolute energy is unchanged.

### 6.5 Energy formulation

Does the EOS energy include formation enthalpy? Today: **no**. `eos0/eos1` are pure Tammann; the phase-change module keeps $q$ outside the hydrodynamic energies and adjusts $\rho E$ by the latent heat when mass changes phase (`Hydro2_PhaseChange.H:48-49`, `Hydro2.cpp` ≈8718).

Because formation energies are constants per species, the two possible conventions are algebraically equivalent:

$$E_k^\text{abs} = E_k^{s} + \sum_i m_{k,i}\,e^\circ_{i},\qquad (\rho E)^\text{abs} = (\rho E)^{s} + \sum_{k,i}m_{k,i}\,e^\circ_{i}.$$

Since species are advected in flux form with fluxes consistent with the phase mass flux, transporting $E^s$ with the present HLLC and species with Section 4 is identical to transporting $E^\text{abs}$.

**Recommendation: keep storing sensible ("hydrodynamic") energies**, as the phase-change module already does, and treat chemistry and phase change the same way:

$$\Delta E_k^{s} = -\sum_i\Delta m_{k,i}\,e^\circ_i,\qquad \Delta(\rho E)^{s} = \sum_k\Delta E_k^{s}.$$

Consequences:

- The Riemann solvers, the Tammann inversion and every `E0/E1` line keep their meaning; with no reactions and no phase change nothing changes.
- Absolute energy is conserved discretely exactly when species mass is; the cell clip of Section 4.1 B1 is then an energy error of size (clipped mass) × $e^\circ$ — one more reason its count must be zero.
- Choice of reference: take $e^\circ_i = e_i(T_0) - c_{v,i}(T_0)\,T_0$ with $T_0$ the lower validity limit of the polynomial (constant-$c_v$ extrapolation to 0 K). Then $e_i^s(T) = e_i(T) - e_i^\circ > 0$ and close to $c_vT$, so the energy-$\gamma$ of Section 3.4 is well conditioned. A 298 K reference would make $e^s \le 0$ for cold gas and must not be used. For stiffened-gas species $e^\circ_i = q_i$.
- The chemistry backend works in absolute energy internally (PelePhysics' constant-volume reactor conserves $\rho$ and $e$); the shim converts on entry and exit.

### 6.6 Operator splitting and the ODE integrator

Per level, per step of size $\Delta t$:

- **Strang**: $\mathcal{C}^{\Delta t/2}\,\mathcal{H}^{\Delta t}\,\mathcal{C}^{\Delta t/2}$ (Strang, SIAM J. Numer. Anal. 5 (1968) 506), where $\mathcal{H}$ is the present RK advance *including* its per-stage mechanical (and phase-change) relaxation, and $\mathcal{C}$ is: gas-phase reaction at constant gas volume → sensible-energy bookkeeping (§6.5) → `RelaxAndReinit` (the reaction raised $p_g$) → optional phase-change relaxation. `chem.splitting = lie | strang` with Lie ($\mathcal{C}^{\Delta t}\mathcal{H}^{\Delta t}$) first, because it is one call site.
- Call site: in `Hydro2::Advance` after the time integrator returns (near the existing `defer_relax` handling, ≈4915) and, for Strang, before the pointer swaps (≈4749). New member `void ReactChemistry(int lev, Set::Scalar dt)`.
- In a mixed cell the reaction is carried out on the gas phase alone at $(\rho_g, T_g, Y_g)$ with $\eta_g$ fixed; the liquid does not react in the first implementation. Cells with $\eta_g <$ `chem.alpha_min`, $T_g <$ `chem.T_min`, or no fuel/oxidiser are skipped.
- **Integrator.** CVODE (variable-order BDF, Newton, dense direct, analytic Jacobian where the backend provides one) for both external backends. With PelePhysics it is the documented path. With Mutation++, which has no integrator, use the same CVODE wrapper with Mutation++ providing $\dot\omega$ (and its Jacobian or a finite-difference one). For the `native` backend, to keep the first tests free of any external library: a built-in adaptive linearly-implicit Rosenbrock or backward-Euler/BDF2 Newton using a small fixed-size LU. Unknowns $(\rho Y_1..\rho Y_N, T)$ at constant $\rho$.
- **Sub-cycling.** CVODE sub-steps internally to its tolerances (`chem.rtol`, `chem.atol`); no explicit sub-cycle loop is needed. Optionally limit the hydrodynamic step by the reaction: $\Delta t \le C_\text{chem}\,\min_\text{cells}\,e_g/|\dot e_\text{chem}|$ (CASTRO has limiters of this kind — M, **TODO verify** names), default off.
- **Stiff-source wave-speed error.** Split reactive Euler solvers on grids that do not resolve the reaction zone can propagate the front at a wrong, grid-dependent speed (LeVeque & Yee, J. Comput. Phys. 86 (1990) 187). This is a property of the splitting, not of this code, and it is the main reason to (i) add a refinement criterion on $|\nabla T|$ or heat-release rate to `TagCellsForRefinement` (≈6088), and (ii) include grid-convergence of the detonation speed in the ZND tests.
- AMR sub-cycling: chemistry runs per level with that level's $\Delta t$ and changes only cell-local state, so reflux is unaffected; `average_down` then overwrites coarse cells as for every other field.

---

## 7. Verification suite

Ordered so that each row gates the next (repository rule: run the short gate before any long or downstream test). Decks under `tests/FlowSpecies/`, checking scripts under `tests/FlowSpecies/reference/` following the pattern of `tests/FlowPhaseChange/reference/*_check.py`. "Accuracy" entries are design targets to be confirmed or revised when the test first runs, not results.

**No reference number is quoted here.** Where a reference comes from a tool, the script that produces it (with tool name and version, mechanism file and its hash) is committed next to the deck and its output stored as a small data file.

### Stage-0 gate (always)

| ID | Test | Checks | Reference | Target | Deck / program |
|---|---|---|---|---|---|
| S0 | Feature off | `species.on = 0` reproduces the baseline | `fcompare` (run as `mpirun -np 1 fcompare`) against plotfiles from the pre-change binary on `FlowRiemannUnitTests/input_Toro1a`, `input_Garrick`, `FlowPhaseChange/input_0D_Equilibrium`, one AMR deck | bit-identical | existing decks |
| S0b | One species per phase, constant $\gamma$ | species path with the trivial composition equals the species-off run | species-off run | round-off ($\le 10^{-12}$ relative) | `input_1D_Toro1a_OneSpecies` |

### Transport (Stage 1)

| ID | Test | Checks | Analytical / reference solution | Target | Deck |
|---|---|---|---|---|---|
| S1 | N passive species, uniform $u$, periodic, smooth (sine) and top-hat $Y_i$ | advection operator, closure | exact translation $Y_i(x - ut)$ | design order of the limiter on the smooth profile; $\max|\sum_i Y_i - 1| \le 10^{-13}$; each $\sum_\text{cells}m_i$ constant to round-off; clip count 0 | `input_1D_SpeciesAdvection` |
| S2 | Trace species at $Y = 10^{-10}$ beside a steep front of the major species | trace preserved, not rescaled or clipped | exact translation; total trace mass constant | trace mass constant to round-off; no negative values; compare majority-closure vs CMA-rescale switch | `input_1D_TraceSpecies` |
| S3 | Species across a moving liquid–gas interface (droplet translating in a gas with a $Y$ gradient) | $Y = m/M$ in trace-phase cells, ghost fill, AMR regrid + reflux | exact translation; species mass conserved with AMR sub-cycling | mass drift round-off with reflux on (this test also exposes the missing `vapor` reflux, §1.6) | `input_2D_SpeciesInterfaceAMR` |
| S4 | `vapor` regression | `H2O` as a gas species reproduces the existing `vapor` field | existing `FlowPhaseChange` decks and their check scripts | round-off in pure transport; solver tolerance with phase change | `input_0D_Equilibrium_Species`, `input_1D_CavitationTube_Species` |

### Multi-species EOS (Stage 2)

| ID | Test | Checks | Analytical / reference solution | Target | Deck |
|---|---|---|---|---|---|
| S5 | Isobaric, iso-velocity contact between two ideal gases with different $\gamma$ and $W$ | pressure-equilibrium preservation inside a phase | $p$ and $u$ stay uniform exactly | report $\max|p - p_0|/p_0$ vs resolution; nonzero is expected for a conservative scheme (Abgrall 1996) — this test **quantifies** it and sets the criterion for when a gas–gas interface must be modelled as two phases | `input_1D_IsobaricSpeciesContact` |
| S6 | Shock tube, two calorically perfect gases, different $\gamma$ | species-dependent EOS through HLLC and reinit | exact two-gas Riemann solution (rarefaction/contact/shock with $\gamma_L \ne \gamma_R$; extend `tests/FlowRiemannUnitTests/Read_exact_Riemann.py`) | first-order convergence in $L_1$; shock and contact positions within one cell | `input_1D_TwoGammaShockTube` |
| S7 | Same shock tube with the two gases as two *phases* (existing capability) | cross-check of S6 | same exact solution | agreement of S6 and S7 within discretisation error | `input_1D_TwoGammaShockTube_Phases` |
| S8 | Isochoric mixing: two gases at different $T$, $p$ in one closed 0-D cell brought to equilibrium | `T_from_e`, mixture rules | closed form for constant $c_v$: $T = \sum m_ic_{v,i}T_i/\sum m_ic_{v,i}$, $p = \rho R_uT/\bar W$ | round-off | `input_0D_IsochoricMixing` |
| S9 | Isobaric mixing by diffusion-free advection of two streams into a constant-pressure region | enthalpy bookkeeping | $T = \sum\dot m_ic_{p,i}T_i/\sum\dot m_ic_{p,i}$ | discretisation error, converging | `input_1D_IsobaricMixing` |

### Relaxation (Stage 3) — stand-alone programs, no flow solve

Built as `src/species_relax_test.cc` → `bin/species_relax_test-2d-g++` (the Makefile builds every `src/*.cc`).

| ID | Test | Checks | Reference | Target |
|---|---|---|---|---|
| R1 | N-phase mechanical relaxation, random admissible states, $N_p = 2, 3, 5$ | generalised scalar solve | (i) $N_p = 2$: present `RelaxAndReinit` kernel; (ii) direct solve of the full system $\{p_k = p\ \forall k,\ \sum\eta_k = 1,\ \text{energy}\}$ by an independent high-precision bracketing solver | $|\Delta p|/p \le 10^{-10}$; iteration count histogram |
| R2 | $p$–$T$ 2×2 Newton | Jacobian, line search | finite-difference Jacobian check; constant-$c_v$ case against `PhaseChange::ptEquilibrium` | Jacobian agreement $10^{-6}$ relative; solutions $10^{-10}$ |
| R3 | $p$–$T$–$\mu$, $M = 1$ | reduction to the present solver | `PhaseChange::relax` on the states of `equilibrium_0d_check.py` and `saturation_table.py` | solver tolerance |
| R4 | $p$–$T$–$\mu$, $M = 2$ (two condensables + one inert) | coupled Newton, active set | Raoult/Dalton flash at given $(T, p)$ solved by Rachford–Rice (closed-form 1-D root), then fed back as a $(V, E)$ problem | $10^{-8}$ in mole fractions |
| R5 | Robustness sweep | fallback path, bounds | every state must return either a converged solution or "unchanged"; residual of returned states | zero NaN, zero bound violations |

### Chemistry (Stages 4–5)

| ID | Test | Checks | Analytical / reference solution | Target | Deck |
|---|---|---|---|---|---|
| C1 | 0-D constant-volume reactor, one-step irreversible Arrhenius $A \to B$, constant $\gamma$, `native` backend | ODE integrator, energy bookkeeping (§6.5), splitting | $T(\lambda) = T_0 + \lambda q/c_v$ exactly; $t(\lambda)$ by quadrature of $d\lambda/dt = k(1-\lambda)e^{-E_a/R_uT(\lambda)}$ (computed in the check script to $10^{-12}$); final $T = T_0 + q/c_v$ | final state to round-off; ignition time to integrator tolerance; total energy conserved to round-off | `input_0D_ConstVolReactor_1step` |
| C2 | 1-D detonation, one-step Arrhenius, constant $\gamma$ | reaction–shock coupling, stiff-source front speed | Chapman–Jouguet: $D_{CJ} = \sqrt{a_0^2 + \tfrac{(\gamma^2-1)q}{2}} + \sqrt{\tfrac{(\gamma^2-1)q}{2}}$, $p_{CJ}/p_0 = (1+\gamma M_{CJ}^2)/(\gamma+1)$, von Neumann $p_{vN}/p_0 = 1 + \tfrac{2\gamma}{\gamma+1}(M_{CJ}^2-1)$; steady ZND profile by integrating the ZND ODE in the check script (Fickett & Davis, *Detonation*) | $D$ within 1 % at ≥ 20 cells per half-reaction length and converging; report the under-resolved behaviour | `input_1D_ZND_1step` |
| C3 | 0-D constant-volume ignition, H$_2$–O$_2$(–Ar/N$_2$), detailed mechanism | PelePhysics thermo/kinetics, unit conversion, CVODE | Cantera `IdealGasReactor` with the **same** mechanism file (the Cantera YAML that CEPTR converts), same initial state; ignition delay by max $dT/dt$ | $T(t)$ and species histories within integrator tolerance; ignition delay within 0.5 % | `input_0D_ConstVolReactor_H2O2` |
| C4 | CJ speed and ZND structure, H$_2$–O$_2$–Ar | full detonation | CJ speed and state from SDToolbox (`CJspeed`, `PostShock_eq`) or NASA CEA (detonation problem; Gordon & McBride, NASA RP-1311); ZND profile from SDToolbox `zndsolve` — all with the same mechanism/thermo file | $D$ within 1 % of CJ once the induction zone is resolved; von Neumann and CJ pressures within 2 %; induction length within 5 % | `input_1D_ZND_H2O2Ar` |
| C5 | Liquid present, no evaporation: C1 run in a cell that is half inert liquid | gas-phase-only reaction + pressure relaxation in a mixed cell | closed system: final $p$, $T_g$ from the $p$-relaxed energy balance (computed by the R1 solver in the check script) | solver tolerance | `input_0D_ReactorWithLiquid` |

### Transport and phase change with species (Stages 6–7)

| ID | Test | Checks | Analytical / reference solution | Target | Deck |
|---|---|---|---|---|---|
| D1 | Binary diffusion of a species step, constant $D$, no flow | species diffusion flux, $\sum\mathbf{j} = 0$ | error-function solution | second order | `input_1D_SpeciesDiffusion` |
| D2 | 1-D laminar premixed flame, H$_2$–air | diffusion + conduction + chemistry | Cantera `FreeFlame` with the same mechanism and transport model | $S_L$ within 3–5 % (transport-model dependent); note a compressible explicit solver needs a long run to settle — use a short domain and NSCBC outflow | `input_1D_PremixedFlame` |
| P1 | 1-D Stefan evaporation: liquid pool evaporating into a non-condensable gas through diffusion | $\Gamma$ with species, interface $Y$ at saturation | classical Stefan-tube solution (interface position $\propto\sqrt t$, logarithmic $Y$ profile) | 5 %, converging | `input_1D_StefanEvaporation` |
| P2 | Single droplet evaporation in hot quiescent gas | the fuel-droplet goal | $d^2$-law: $d^2(t) = d_0^2 - Kt$, $K = 8\rho_gD\ln(1+B_M)/\rho_\ell$ (the code already has a Spalding-number path, `Hydro2.cpp` ≈3567) | slope $K$ within 10 % (the law's own assumptions are that loose). Needs spherical symmetry: 3-D octant, or an axisymmetric/spherical source — I found no axisymmetric option in `Hydro2.cpp`, **TODO verify** | `input_3D_DropletD2Law` |

### Plasma (Stage 8)

| ID | Test | Checks | Analytical / reference solution | Target | Deck |
|---|---|---|---|---|---|
| I1 | 0-D ionisation equilibrium of a monatomic gas (H or Ar, single ionisation) at given $T, p$ | electrons as a species, charge neutrality, equilibrium composition | Saha: $\dfrac{\alpha^2}{1-\alpha^2} = \dfrac{k_BT}{p}\,\dfrac{2g_+}{g_0}\Big(\dfrac{2\pi m_ek_BT}{h^2}\Big)^{3/2}e^{-E_\text{ion}/k_BT}$ | within the accuracy of the partition-function data used by the library (percent level); state which statistical weights are used | `input_0D_SahaIonisation` |
| I2 | Two-temperature relaxation in a 0-D box ($T_e \ne T_h$ initially) | second gas energy, energy-transfer source | exponential relaxation with the library's own relaxation time (self-consistency), energy conserved | round-off in energy | `input_0D_TwoTemperatureRelax` |

---

## 8. Staged implementation plan

Rules for every stage:

- New input switch, **default off**; with it off the binary is bit-identical to the previous one on the S0 deck list, checked with `fcompare`.
- After editing any header, `touch src/hydro2.cc` before building (known stale-dependency trap in this Makefile).
- Run the stage's short gate test before any longer test; one new test at a time.
- No file deletion; scratch builds and outputs go to fresh directories.

Effort is one person already familiar with `Hydro2.cpp`, in working weeks, and is a rough estimate.

| Stage | Content | Switch | Gate tests | Effort |
|---|---|---|---|---|
| 0 | Decisions (Section 8.2). Add `tests/FlowSpecies/` skeleton and the S0 baseline plotfiles. Restore or rewrite the missing `bin/PeleC.md` content into `docs/`. | — | S0 | 0.5 |
| 1 | `Hydro2_Species.H`: parse (`species.on`, `species.gas.names`, `species.liquid.names`), field + old + diag, RK slot `ix_sp`, RHS row (flux = phase mass flux × face $Y$ with majority closure, §4), cell closure, ghost fill by $Y$, IC by $Y$ expressions, reflux components, post-`FillPatch` closure. Passive only — EOS unchanged. Fix the `vapor` reflux at the same time. | `species.on` | S0, S0b, S1, S2, S3 | 2 |
| 2 | Native multi-species thermodynamics (constant-$c_p$ species: $\gamma_i, W_i, c_{v,i}, h_{f,i}$; stiffened-gas liquid species), phase temperatures with one definition, per-cell $(\hat\gamma,\hat\pi)$ cache, replacement of the `eosK.Gamma()/P0()` reads in `RHS`, `FillGhost4BC`, `RelaxAndReinit` (true-EOS reinit), `PostAverageDown`, NSCBC species waves. | `species.eos = fixed \| mixture` | S5, S6, S7, S8, S9 | 3 |
| 3 | `Solver/Local/Relax/Newton.H` (scaled Newton, line search, active set, fallback); $p$–$T$ and $p$–$T$–$\mu$ for species; phase change re-expressed on species with the present solver as fallback and reference; `vapor` path kept until S4 passes, then retired behind a compatibility name. | `phasechange.solver = legacy \| newton` | R1–R5, S4 | 3 |
| 4 | `Solver::Chem` interface, `native` backend (one-step Arrhenius), built-in implicit integrator, `ReactChemistry`, Lie then Strang, sensible-energy bookkeeping, refinement criterion on heat release. | `chem.on`, `chem.backend = native` | C1, C5, C2 | 2 |
| 5 | PelePhysics backend: pin and fetch SUNDIALS + PelePhysics, Makefile mechanism selection, generated `mechanism.cpp` in the build, Fuego EOS shim with SI↔CGS, CVODE wrapper with analytic Jacobian. Install Cantera + SDToolbox for references. | `chem.backend = pelephysics` (build with `PELE_ENABLED`) | C3, C4 | 3–4 |
| 6 | Species diffusion, mixture viscosity/conductivity from the backend, enthalpy diffusion term in the energy rows. | `species.diffusion` | D1, D2 | 2–3 |
| 7 | Evaporation of a multi-component fuel into a reacting gas: $\Gamma$ from Stage 3 feeding Stage 5; liquid species thermodynamic data for the chosen fuel. | uses the above | P1, P2 | 2–3 |
| 8 | Plasma: Mutation++ backend, electrons and ions as gas species, charge neutrality, optional second gas energy with energy-transfer source, ionised transport. Research-grade; scope to be fixed after Stage 5. | `chem.backend = mutationpp` (build with `MPP_ENABLED`) | I1, I2 | 4–6 |
| 9 (optional) | $N_p > 2$: turn `rho_eta0/1`, `energy0/1`, `eta` into phase-indexed multi-component fields and the Riemann/limiter structs into arrays. This is the 1331-line refactor; do it only if a third immiscible medium is required. | — | all two-phase regression decks bit-identical | 6–8 |

Stages 1–5 (about 13–15 weeks) reach "a detonation wave in H$_2$–O$_2$–Ar verified against CJ/ZND". Stage 6 is independent of 5 and can be reordered. Stage 3 is not required for gaseous detonation and can move after Stage 5 if the detonation result is wanted first.

### 8.1 Risks

1. **Pressure oscillations at gas–gas species contacts** (S5). Inherent to conservative mass-fraction formulations. If S5 shows them to be unacceptable for an intended case, that case needs the two gases as separate phases (option c1) — a modelling decision per problem, not a bug to chase.
2. **Local stiffened-gas fit (§3.4) is my construction.** It is exact at the cell state and for constant-$\gamma$ species, but its behaviour through strong shocks with large $\gamma$ variation (dissociating products) is unverified. S6/S7 and C4 test it. The fallback is the Colella–Glaz two-$\gamma$ form, which needs one more member in `State` and edits in every Riemann solver.
3. **$Y = m/M$ in trace-phase cells.** The same ill-conditioning the code has fought for $p_k = (\gamma-1)E_k/\eta_k$ (`Hydro2.cpp` ≈8300–8385, `Limiter.H:130-162`). Expect to need a floor-and-extend rule; S3 is designed to expose it.
4. **Stiff-source front speed** on under-resolved grids (LeVeque & Yee 1990). Needs AMR tagging on the reaction zone and a convergence study in C2/C4.
5. **Build coupling.** PelePhysics has no library target, its `EOS.H` collides by name with Hydro2's, the current head did not compile against the AMReX in `ext/` in one place (`PelePhysicsEOS.H:60-68`), and its reactors may require AMReX built with SUNDIALS. Mitigation: pin versions; use thermo/kinetics only and own the CVODE call.
6. **Unit systems.** PelePhysics CGS vs Hydro2 SI; a single conversion layer and test C3 catch errors here.
7. **Two temperature definitions** already in the code (§1.4). Chemistry is exponentially sensitive to $T$; the species EOS must define $T_k$ once and the plotted `T` should follow it when `species.on`.
8. **Energy consistency under clipping** (§6.5): any species mass created by a positivity clip is also created formation energy. Diagnostics must report it; tests require zero.
9. **`Hydro2.cpp` is a single 9000-line file under active edit** by other work (elastic solid, ablation). Keeping species code in its own header with a handful of small, clearly marked hook sites limits merge conflicts.
10. **Cost.** Detailed chemistry can dominate the run time by a large factor; the machine's per-session core/RAM budget should be checked against a small C4 run before any 2-D detonation deck is planned.

### 8.2 Open questions for the user

1. **Interpretation of "eta as m × n × s".** This document reads it as: `eta` stays the single liquid/gas volume fraction; the new `ncomp = s` field is the species partial density per phase; species volume fractions are a derived plot field. If the intent was instead N immiscible fluids each with its own volume fraction (option a), the plan changes to Stage 9 first. Which is it?
2. **First target mixture and mechanism.** H$_2$–O$_2$–Ar with a 9-species mechanism is assumed. Is a hydrocarbon or a liquid fuel (which one?) needed early? That decides the liquid species data and `HYDRO2_NSPEC_MAX`.
3. **Gaseous or condensed-phase detonation?** Section 7 assumes gaseous. A condensed explosive needs a products EOS (JWL or similar) and a different reaction model (cf. Petitpas, Saurel, Franquet & Chinnayya, Shock Waves 19 (2009) 377 — **TODO verify**), and changes Stage 2.
4. **Libraries.** Agree to PelePhysics first (scaffold exists) and Mutation++ only for the plasma stage? May SUNDIALS, PelePhysics, Cantera and SDToolbox be downloaded into `ext/` / a Python environment, and which versions should be pinned? Is LGPL (Mutation++) acceptable?
5. **Where is `bin/PeleC.md`?** Three files point to it as the integration roadmap; it is not on disk.
6. **Energy convention.** Keep sensible energies in $E_k$, $\rho E$ and book formation energy at reaction/phase change (recommended, matches the phase-change module), or switch the stored energies to absolute?
7. **Temperature in mixed cells.** Should chemistry see the gas-phase temperature $T_g$ (recommended) or a cell temperature after liquid–gas thermal relaxation?
8. **Retire `vapor`?** Stage 3 replaces it by a gas species named in the deck; plot names and the `phasechange.Yv0` input in the `FlowPhaseChange` decks would change. Acceptable, or keep both paths?
9. **Majority closure vs proportional rescaling** as the default face rule (§4.1). The document adopts the user's rule with rescaling as the fallback; S2 will compare them. Confirm.
10. **Is GPU execution ever a goal?** It affects whether Mutation++ can sit in the main path and whether the virtual limiter/Riemann dispatch should be addressed.
11. **Transport.** Is species diffusion (flame-speed test, Stage 6) wanted before plasma, or is inviscid detonation the only near-term chemistry target?
12. **Geometry for the $d^2$-law test.** Is an axisymmetric or spherically symmetric mode available or planned, or should P2 be a 3-D octant?
13. **Third immiscible medium.** Is there a planned case with liquid + gas + deforming solid sharing cells? If yes, Stage 9 moves up and the phase-indexed layout should be adopted from the start.

---

## References

Checked bibliographically from memory; items marked **TODO verify** need their details confirmed before being cited elsewhere.

- G. Allaire, S. Clerc, S. Kokh, J. Comput. Phys. 181 (2002) 577.
- R. Abgrall, J. Comput. Phys. 125 (1996) 150.
- S. H. Bryngelson et al., "MFC: An open-source high-order multi-component, multi-phase, and multi-scale compressible flow solver", Comput. Phys. Commun. 266 (2021) 107396.
- A. Chiapolino, P. Boivin, R. Saurel, "A simple and fast phase transition relaxation solver for compressible multicomponent two-phase flows", Comput. Fluids 150 (2017) 31. **TODO verify**
- P. Colella, H. M. Glaz, "Efficient solution algorithms for the Riemann problem for real gases", J. Comput. Phys. 59 (1985) 264.
- W. Fickett, W. C. Davis, *Detonation*, University of California Press (1979).
- S. Gordon, B. J. McBride, NASA Reference Publication 1311 (1994) — NASA CEA.
- M. T. Henry de Frahan et al., "PeleC: An adaptive mesh refinement solver for compressible reacting flows", Int. J. High Perform. Comput. Appl. (2023). **TODO verify**
- A. K. Kapila, R. Menikoff, J. B. Bdzil, S. F. Son, D. S. Stewart, Phys. Fluids 13 (2001) 3002.
- S. Karni, J. Comput. Phys. 112 (1994) 31.
- O. Le Métayer, J. Massoni, R. Saurel, Int. J. Therm. Sci. 43 (2004) 265.
- O. Le Métayer, J. Massoni, R. Saurel, "Dynamic relaxation processes in compressible multiphase flows. Application to evaporation phenomena", ESAIM: Proc. 40 (2013) 103. **TODO verify**
- O. Le Métayer, R. Saurel, "The Noble-Abel Stiffened-Gas equation of state", Phys. Fluids 28 (2016) 046102.
- R. J. LeVeque, H. C. Yee, J. Comput. Phys. 86 (1990) 187.
- M. Pelanti, K.-M. Shyue, J. Comput. Phys. 259 (2014) 331.
- M. Pelanti, K.-M. Shyue, "A numerical model for multiphase liquid–vapor–gas flows with interfaces and cavitation", Int. J. Multiphase Flow 113 (2019) 208. **TODO verify**
- F. Petitpas, R. Saurel, E. Franquet, A. Chinnayya, Shock Waves 19 (2009) 377. **TODO verify**
- T. Plewa, E. Müller, "The consistent multi-fluid advection method", Astron. Astrophys. 342 (1999) 179.
- R. Saurel, F. Petitpas, R. Abgrall, J. Fluid Mech. 607 (2008) 313.
- R. Saurel, F. Petitpas, R. A. Berry, J. Comput. Phys. 228 (2009) 1678.
- K. Schmidmayer, S. H. Bryngelson, T. Colonius, J. Comput. Phys. 402 (2020) 109080.
- K. Schmidmayer, F. Petitpas, S. Le Martelot, É. Daniel, "ECOGEN: An open-source tool for multiphase, compressible, multiphysics flows", Comput. Phys. Commun. 251 (2020) 107093.
- J. B. Scoggins, V. Leroy, G. Bellas-Chatzigeorgis, B. Dias, T. E. Magin, "Mutation++: MUlticomponent Thermodynamic And Transport properties for IONized gases in C++", SoftwareX 12 (2020) 100575 (arXiv:2002.01783).
- G. Strang, SIAM J. Numer. Anal. 5 (1968) 506.
- Web pages opened on 2026-10-06: `github.com/AMReX-Combustion/PelePhysics` (README, `license.txt`), `pelephysics.readthedocs.io` (Introduction, "CVODE implementation in PelePhysics"), `github.com/mutationpp/Mutationpp` (README), `arxiv.org/abs/2002.01783`.

## Appendix: files created by this investigation

- `docs/design/Hydro2_MultiSpecies_Chemistry_Design.md` (this file).
- `bin/tests/species_scratch/count2phase.sh` — read-only script that reproduces the table in Section 1.9.
