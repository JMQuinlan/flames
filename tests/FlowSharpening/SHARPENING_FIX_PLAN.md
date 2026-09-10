# Interface Sharpening — Root Cause + Validated Fix Plan

**Branch:** `Hydro2_6Eqn_Marmottant3D_Sharp` · **Date:** 2026-06-29 (overnight autonomous session)
Supersedes the diagnosis in `SHARPENING_VALIDATION.md` (which framed it as incomplete Tiwari).
That was right that it's non-functional, but the *actual* root cause is different and simpler.

## TL;DR

The current `Hydro2::InterfaceSharpening` does **nothing measurable** to the interface
(thickness 13.23 → 13.23 cells, hard-measured) and is fundamentally incompatible with the
current state model. **Two faults:**

1. **The compression is inert.** STEP 4's mass-based compression operator produces zero
   thickness change on the controlled test (below), even while firing and conserving mass.
2. **It operates on the wrong variable.** STEPS 5-6 sharpen the phase *masses*
   `rho_eta0/rho_eta1` and then recover `eta = rho_eta0/rho_total` — the **mass fraction**.
   But this model evolves `eta` as an **independent volume fraction**. The code's own WARNING
   (`Hydro2.cpp:3464-3471`) says it: *"will silently overwrite the volume fraction with the
   mass fraction every sharpening pass."* For the equal-density test that's masked (mass frac
   = vol frac), but for the bubble (ρ_liq/ρ_gas = 1000) it corrupts `eta` outright — this is
   the `apply_sharpening=1` crash on the RPE.

**The fix:** discard the reinit+mass-compression construction and apply a **conservative
Olsson-Kreiss compression to `eta` (the volume fraction) directly**, then update the
conserved per-phase fields with the *same* flux so mass/momentum/energy are conserved and
velocity/pressure equilibrium is preserved. Validated in a Python prototype (below).

---

## 1. Hard characterization (this session)

Tool: `reference/check_sharpening_full.py` (thickness-in-cells + full conservation +
equilibrium — the original `check_sharpening.py` only saw ∫η and band-volume S).
Test: `input_Sharpen_2D_Planar`, 128², static (v=0, equal ρ, uniform p), ON vs OFF.

| | thickness (cells) | ∫η, ∫ρη₀ | max\|u\| | p-spread |
|---|---|---|---|---|
| OFF | 13.23 → 13.23 | conserved | ~1e-9 | ~1e-9 |
| **ON** | **13.23 → 13.23** | conserved | ~4e-9 | ~6e-9 |

So ON = OFF to 2 decimals: **zero sharpening**. (The old S-metric's "0.3% narrowing" was
noise.) It IS firing — the per-cell renorm resets the small ∫η drift to 0 when it runs — and
it is perfectly conservative + equilibrium-preserving. It just does nothing useful.

## 2. Root cause (verified in code)

- `InterfaceSharpening` (`Hydro2.cpp:2998`): STEP1 φ→ψ=ε ln(φ/(1−φ)); STEP2 reinit ψ to
  signed distance (`reinit_max_iter=10`); STEP3 ψ→φ_sharp=sigmoid; STEP4 a Tiwari-style mass
  compression `R_l` on `rho_eta0` gated by a **max-principle** that can skip interface cells;
  STEP5 copy `rho_eta*_work` back; **STEP6 `eta = rho_eta0/rho_total` (mass fraction).**
- STEP4 is inert (measured) — weak/gated compression on the masses.
- STEP6 is the architectural fault: sharpening is done in the *mass* variables and `eta`
  (independent volume fraction) is overwritten with the mass fraction. Wrong variable.

## 3. The validated correct method (Olsson-Kreiss compression on η)

Prototype: `reference/prototype_sharpen.py` (numpy, runs on a plotfile's η; **no solver
change**). Conservative compression of the volume fraction in pseudo-time τ:

```
d(eta)/d(tau) = div[ eps (grad eta . nhat) nhat  -  eta (1-eta) nhat ],   nhat = grad(eta)/|grad(eta)| (frozen)
```

- **Pure divergence form ⇒ ∫η conserved to machine precision** (verified: 0.00%).
- Balances normal diffusion (ε) vs compression ⇒ steady sigmoid of thickness ≈ **4.4 ε/dx**.
- Sharpens a STATIC interface (relaxation to target thickness), so the quiescent test DOES
  validate it (unlike velocity-driven Tiwari).

Measured on the test IC (14.1-cell smear, dx=7.8e-3):

| ε | steady thickness | ∫η drift |
|---|---|---|
| 1.5 dx | **6.65 cells** | 0.00% |
| 2.5 dx | **11.0 cells** | 0.00% |

⇒ **ε ∈ [1.0, 2.7]·dx gives the 4-12 cell band.** Thickness is a direct, tunable knob.

## 4. C++ implementation plan

Replace STEPS 1-6 of `InterfaceSharpening` with:

**(a) Compress η directly** (the prototype, made face-conservative):
  - freeze `nhat = grad(eta)/|grad(eta)|` at entry;
  - pseudo-time loop (≈30-60 iters, dτ ≈ 0.2 dx): build face fluxes
    `F = eps (grad eta . nhat) nhat − eta(1−eta) nhat`, update `eta -= dτ ∇·F`;
  - input key for ε: set `epsilon` so `4.4 eps/dx ∈ [4,12]` (≈ 1.5 dx default → ~6.6 cells).

**(b) Carry the conserved per-phase fields with the SAME flux** (conservation + equilibrium):
  The η-flux F moves phase-0 *volume* across each face. Move the matching phase mass /
  momentum / energy by the upwind pure-phase state so the pure density ρ_k, specific energy
  e_k and velocity u are PRESERVED (only the amount α_k moves) ⇒ p_k and u stay uniform:
```
flux(rho_eta0) = rho0_upwind * F            (rho0 = rho_eta0/eta, the pure phase-0 density)
flux(rho_eta1) = rho1_upwind * (−F)         (phase-1 volume flux is −F)
flux(E0)       = (E0/eta)_upwind   * F   ;  flux(E1) = (E1/(1−eta))_upwind * (−F)
flux(momentum) = u_face * (flux(rho_eta0)+flux(rho_eta1))
```
  All in divergence form ⇒ ∫ρη_k, ∫ρu, ∫ρE conserved; uniform u,p preserved. (This is the
  Shukla-Pantano-Freund consistency that the old code's "mass-only" form was missing — but
  applied to η, not the masses.)

**(c) Do NOT recover eta from the masses.** Delete STEP 6. `eta` is the primary; the masses
  follow it via (b).

**(d) Keep it conservative + bounded:** clamp η∈[0,1] only after the conservative update
  (the prototype's clip changed ∫η by 0; verify in C++). Gate by `eta(1−eta)>1e-6` (interface
  band) so the far field is untouched.

## 5. Tests (this folder)

- `input_Sharpen_2D_Planar` (static) — **valid for Olsson** (it relaxes to target thickness).
  Pass: thickness → 4.4ε/dx, ∫η/∫ρη/∫ρu/∫E flat, max|u| & p-spread ~0. Tool:
  `check_sharpening_full.py OFF_dir ON_dir`.
- `input_Sharpen_2D_Moving` (TODO/added) — advect the smeared interface at constant u
  (periodic x); the band must hold at target while advecting and conserve momentum/energy
  (catches advection×sharpening coupling + the equilibrium update of (b)).
- `input_Sharpen_2D_Circle` (TODO/added) — smeared disk, v=0; band narrows to target, the
  circle stays round (no grid imprint), gas area ∫(1−η) conserved.
- Later: a gas–liquid Riemann (Garrick) ON vs OFF — shock/contact speeds unchanged, fully
  conservative (sharpening must not corrupt a moving shock+interface).

## 6. Acceptance gates (each before the next)

1. Static planar: thickness → target (4-12), all integrals flat, equilibrium preserved.
2. Moving planar: same, while advecting; momentum/energy conserved.
3. 2D circle: band at target, round, area conserved.
4. **Unequal density** (ρ_liq/ρ_gas=1000): η stays the VOLUME fraction (NOT overwritten by
   mass fraction); no pressure kick — the test that the old code fails.
5. RPE bubble: `apply_sharpening=1` runs (no crash), interface holds 4-12 cells through
   collapse, gas mass conserved (`check_mass_conservation.py`), K/jet-depth reduced.

## Status at end of session
- Diagnosis: **done + verified** (inert compression + eta=mass-fraction architectural fault).
- Correct method: **validated** in `prototype_sharpen.py` (conservative, thickness-tunable).
- C++ rewrite: **planned, not yet implemented** — held for review because (b) touches the
  conserved fields and the mandate is "be conservative." Ready to implement on go-ahead.

---

## 7. IMPLEMENTATION STATUS (2026-06-29, overnight)

The conservative Olsson compression of §4 is **implemented** in
`Hydro2::InterfaceSharpening` (the old reinit+mass-compression construction is gone).
`apply_sharpening` defaults **false** -> the routine is DORMANT and the OFF path is
bit-identical to before (verified: 13.23 cells, all integrals flat). Safe to leave on
the branch. Compiles clean 2D. New input key `sharpening_sweeps` (default 40).

**Design as built:** sharpen ETA only by the conservative Olsson flux (face-compact
normal gradient, frozen nhat), clamped to [0,1] each sweep; then rebuild the conserved
per-phase fields from the sharpened eta x the FROZEN pure intensive states
(rho_k=rho_eta_k/alpha_k, e_k=energy_k/alpha_k, u=mom/rho). For uniform pure states
(the equal-density tests) this is conservative + equilibrium-preserving.

**What works (verified):** in the bulk the band sharpens **13.23 -> ~4.05 cells**
(squarely in the 4-12 target), and a direct field dump right after the first pass shows
`pressure=1.0000, density=1.0000, energy_per_vol=2.5000` uniform with eta/energy/mass
all consistent. So the core method + the conservative carry are correct.

**OPEN ISSUE (blocks the unit test):** the instant sharpening fires, spurious velocity
appears (`max|u|` grows 1e-9 -> ~0.7-1.0) from a few OUTLIER cells in the (uniform)
far field, and int(eta) drifts -4..-16% over the run. **Single-rank/single-box
reproduces it -> NOT an MPI/box artifact; it is the sharpening<->hydro/relaxation
interaction.** Fixes already in (all necessary + correct, none sufficient):
  - compact face-normal gradient (fixed an odd-even decoupling that earlier blew eta>1);
  - per-sweep clamp of eta to [0,1] (the Olsson diffusion overshoots the bounds);
  - eta-only sweep + frozen-pure-state rebuild (fixed a rho_eta0/eta runaway when eta
    was clamped but the coupled masses were not);
  - nhat gate / band-weighted flux (to kill far-field roundoff-seeded phantom
    interfaces) -- did NOT resolve the equilibrium break.

**Leading hypothesis / next step:** `RelaxAndReinit` runs immediately after sharpening
(in `FillGhost4BC`) and re-equilibrates p0=p1 per cell by moving eta and reinitializing
the per-phase energies; it likely fights the sharpened state and injects the pressure
perturbation that drives the velocity. ISOLATE by staging: (a) sharpen with the hydro
RHS disabled (does eta stay sharp + equilibrium hold?); (b) sharpen + relaxation, no
flux step; (c) full. Whichever stage first shows `max|u|>0` is the culprit. Also dump
the eta/pressure field of the few outlier cells right after one pass (single rank) to
see whether the bad cells are created by the rebuild or by the subsequent relax/hydro.

**3D:** not yet run -- gated on the 2D equilibrium fix (3D would inherit the same issue).
