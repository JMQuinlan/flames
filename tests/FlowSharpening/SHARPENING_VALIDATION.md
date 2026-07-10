# Interface-Sharpening Validation & Diagnosis

**Branch:** `Hydro2_6Eqn_InterSharp`  ·  **Date:** 2026-06-26

## TL;DR

The interface sharpening (`apply_sharpening=1`, `Hydro2::InterfaceSharpening`) **fires but does
not sharpen**. A controlled 2D test shows it is *mass-conservative* but the interface band does
**not** narrow — it stays put (in fact widens ~0.4%). Root cause: the routine is an **incomplete
and partly sign-inverted implementation of Tiwari, Freund & Pantano (2013)** — it regularizes only
the phase *masses*, omits the momentum and energy source terms the method requires, uses a fixed
pseudo-timestep instead of the velocity-driven rate, and the compression update appears sign-flipped.

---

## 1. Test (`input_Sharpen_2D_Planar` + `reference/check_sharpening.py`)

A deliberately over-smeared planar interface (`eta = ½(1+tanh((x−0.5)/0.05))`, band ≈ 19 cells) in a
**quiescent, uniform-pressure, equal-density** box (`v=0`, uniform `p`, `ρ0=ρ1=1`, `σ=0`). With those
ICs the hydro **cannot move the interface**, so any change in `eta` is due *only* to sharpening.
The current sharpening is velocity-independent (`dt_compression = 0.5 h²`), so a static test is valid
for it. Run `apply_sharpening=0` vs `1` (128², uniform, sharpening every 2 steps) and measure two
global integrals (AMR-correct):

- **M = ∫η dV** — conserved volume fraction (must stay flat)
- **S = ∫η(1−η) dV** — "band volume": shrinks as the interface sharpens (no ray needed)

### Result

**Original code (`-=`), 2D:** ∫η conserved to 0.0000%, but **S/S₀ = 1.0040 → the band WIDENS**
(anti-sharpening). Sharpening fired 27× — active, but inverted.

**After the sign fix (`+=`, see §4), 2D and 3D (`input_Sharpen_3D_Planar`, 64³):**

| case | S/S₀ (ON) | S/S₀ (OFF) | max \|ΔM/M₀\| | verdict |
|---|---|---|---|---|
| 2D planar | 0.9962 | 1.0000 | 0.0000% | CONSERVATIVE ✅ · barely sharpens ❌ |
| 3D planar | 0.9970 | 1.0000 | 0.0001% | CONSERVATIVE ✅ · barely sharpens ❌ |

So in **both 2D and 3D**: mass/volume-fraction is conserved to machine precision, but the interface
narrows only **~0.3%** where a 19→3-cell sharpening should show **~50-80%**. The sharpening is
effectively non-functional.

---

## 2. Diagnosis vs the paper

The reference is **Tiwari, Freund & Pantano (2013)** (the φ↔ψ sigmoid `ψ=ε ln(φ/(1−φ))`, the
"Eq. 6/10/15-16/20a" comments, and the bubble-collapse-on-Cartesian-mesh demo all trace there).
The current `InterfaceSharpening` deviates from it in four substantive ways:

1. **Mass-only.** Tiwari adds the regularization `R(α₂)` to **all** conservative equations — mass
   (Eq 30a/30b), **momentum (Eq 30c)** and **energy (Eq 30d)** — which is the paper's central thesis
   (vs Shukla/So): *"conserves mass, momentum, energy and preserves velocity/pressure equilibrium."*
   The code modifies only `rho_eta0/rho_eta1` (mass). With stale momentum/energy, a sharpened `η`
   produces a pressure kick that breaks the equilibrium the method is supposed to preserve.
2. **No velocity-driven rate.** Tiwari's `R = L(α₂)·U₀·n̂·∇(ε|∇α₂| − α₂(1−α₂))` (Eq 20) with
   `U₀ = ‖u_I‖_max = 4(α₂(1−α₂)|u|)_max` (Eq 23). The code uses a fixed `dt_compression = 0.5 h²`
   (`Hydro2.cpp:2976`) — not the paper's rate, and not consistent with the RHS-source formulation.
3. **Sign.** Eq 30a/30b put `R̂` on the RHS as a *source* (`∂(ρα)/∂t = +R̂`), so a forward update
   **adds** it. The code subtracts: `rho_eta0 -= ω·dt·R_l` (`Hydro2.cpp:~3225`). That is anti-
   sharpening, matching the observed ~0.4% *widening*.
4. **Ad-hoc per-cell renorm.** The "enforce exact mass conservation" block rescales `ρη0,ρη1` so the
   *mixture* density per cell equals the original. That keeps ∫η conserved (hence the PASS above) but
   is not part of Tiwari and masks the lack of a true conservative (flux-divergence) form.

A `max-principle` gate (`Hydro2.cpp:3059`) and a reinit-to-signed-distance step are layered on top;
the overall scheme is a reinit/pseudo-time construction, **not** the Tiwari RHS-source method.

---

## 3. The correct method (for the fix)

Implement Tiwari's quasi-conservative regularized system (their Eq 29-30), adapted to the 6-eq fields:

```
∂(ρ1α1)/∂t + ∇·(ρ1α1 u)          = R̂1
∂(ρ2α2)/∂t + ∇·(ρ2α2 u)          = R̂2
∂(ρu)/∂t   + ∇·(ρu⊗u) + ∇p       = u·R̂            (R̂ = R̂1 + R̂2)     <-- MISSING in code
∂(ρE)/∂t   + ∇·((ρE+p)u)         = κ R̂ + (p(Γ2−Γ1)+Π2−Π1) R       <-- MISSING in code
∂α2/∂t     + u·∇α2 = α1α2(...)∇·u + R
```
with (Eq 29):
```
R̂2 ≈ L(α2) U0 n̂·( ∇(ε n̂·∇(ρ2α2)) − (1−2α2)∇(ρ2α2) )
R̂1 ≈ L(α2) U0 n̂·( ∇(ε n̂·∇(ρ1α1)) − (1−2α2)∇(ρ1α1) )
U0 = 4 (α2(1−α2)|u|)_max,   L = 1 on 1e-6<α2<1−1e-6 else 0,   n̂ = ∇α2/|∇α2|
```
Key points: (a) it is a **source added to the existing RHS each step** (not a separate reinit pass),
(b) the **same `R`/`R̂` must hit momentum + energy**, (c) `U0 ∝ |u|` so it sharpens *as the flow
moves the interface* (a faithful version will NOT sharpen a static `v=0` interface — so the proper
acceptance test is a **moving** interface, see §5), (d) discretize `R` conservatively (Eq 32-34).

---

## 4. Sign-flip experiment (record)

As the cheapest single hypothesis, the STEP-4 update sign was flipped `-= → +=`
(`Hydro2.cpp:~3225`) and the ON case re-run.

| | S/S₀ at end | meaning |
|---|---|---|
| original `-=` | 1.0040 | band **widens** (anti-sharpening) |
| flipped `+=`  | 0.9962 | band **narrows** — correct direction |

**Conclusion:** the sign was indeed inverted (direction flips from widening to narrowing), so `+=` is
kept as a partial correction. But the magnitude (0.38% narrowing) is **negligible** vs the ~50-80%
the test should show, confirming the sign was *necessary but far from sufficient*. The scheme is too
weak/incomplete to sharpen — consistent with the missing momentum/energy source terms and the absent
`U₀ ∝ |u|` rate. **A faithful Tiwari re-implementation (§3) is required; a sign fix alone is not.**

---

## 5. Recommended acceptance tests (a "few", 2D + 3D + Riemann)

- **Static planar (this file):** valid only for a velocity-independent sharpener; good regression for
  "does it conserve and not blow up." A *faithful* Tiwari sharpener correctly does ~nothing here.
- **Moving planar (TODO):** advect a smeared interface at constant `u`; the band must stay at the
  target thickness (sharpen toward `ε`) while ∫η and momentum/energy are conserved. This is the real
  test of the velocity-driven Tiwari `R`.
- **3D sphere (TODO):** smeared bubble; check ∫(1−η) (gas volume) conserved + band narrows, and the
  bubble stays spherical (Tiwari Fig 2 — symmetry on a Cartesian mesh).
- **Garrick gas-liquid Riemann (TODO):** sharpening ON vs OFF; the shock/contact speeds and the
  exact-solution match must be unchanged and mass/momentum/energy conserved (sharpening must not
  corrupt a moving shock+interface).

---

## References (in `flames_PAPERS/InterfaceRelax/`)

1. **A. Tiwari, J. B. Freund, C. Pantano (2013).** *A diffuse interface model with immiscibility
   preservation.* J. Comput. Phys. **252**, 290-309. doi:10.1016/j.jcp.2013.06.021.
   — file: `tiwari-freund-pantano-jcp-2013.pdf`. **The sharpening method (Eq 13/20/29/30).**
2. **R. Saurel, F. Petitpas, R. A. Berry (2009).** *Simple and efficient relaxation methods for
   interfaces separating compressible fluids, cavitating flows and shocks in multiphase mixtures.*
   J. Comput. Phys. **228**, 1678-1712. doi:10.1016/j.jcp.2008.11.002.
   — file: `1-s2.0-S0021999108005895-main (1).pdf`. The 6-eq model + pressure relaxation ("Sau09").
3. **K. Schmidmayer, S. H. Bryngelson, T. Colonius (2020).** *An assessment of multicomponent flow
   models and interface capturing schemes for spherical bubble dynamics.* J. Comput. Phys. **402**,
   109080. doi:10.1016/j.jcp.2019.109080. — file: `Schmidmayer_2020.pdf`. §5.4 sharpening context.
