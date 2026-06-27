# Marmottant Coated-Bubble Surface Tension

**Branch:** `Hydro2_6Eqn_Marmottant3D`  ·  **Date:** 2026-06-26

A toggleable **add-on** to the existing CSF surface-tension term: when `marmottant=1`, the constant
`sigma` is replaced by the Marmottant (2005) piecewise effective tension `σ(R)`, with the bubble
radius `R` extracted from the **interface-averaged curvature**.

---

## 1. The model (Marmottant 2005, Eq. 4)

The lipid-monolayer shell of a contrast microbubble gives a radius-dependent effective tension:

```
          ┌ 0                         R ≤ R_buckling        (buckled, monolayer crumpled)
 σ(R) =   ┤ χ (R²/R_buckling² − 1)    R_buckling ≤ R ≤ R_break-up   (elastic)
          └ σ_water                   R ≥ R_break-up         (ruptured, bare interface)
```
with `R_break-up = R_buckling·√(1 + σ_break-up/χ)`. Three shell parameters:
- **`R_buckling`** — radius below which the shell buckles and σ→0.
- **`χ`** — shell elastic modulus [N/m] (slope of the elastic regime).
- **`σ_break-up`** — critical tension at which the shell ruptures (≥ σ_water).

(Implemented statelessly: σ jumps to σ_water once the elastic value would exceed σ_break-up. The
*hysteresis* of rupture — staying ruptured if R later shrinks — is a noted future extension.)

## 2. Pseudo-radius from curvature (the part with no single source paper)

The Marmottant tension is a function of the **bubble radius** — a global property of the shell, not a
local quantity. The CSF interface already carries a curvature field. For a sphere/circle the mean
curvature equals the **divergence of the unit normal**, `κ = ∇·n̂` (Brackbill et al. 1992), and for a
`d`-sphere of radius `R` this is the standard differential-geometry identity

```
κ = (d − 1) / R        (2D circle: 1/R ; 3D sphere: 2/R)   ⇒   R = (d − 1) / |κ|.
```

We evaluate it on the **interface-averaged** curvature rather than pointwise:

```
R = (d−1)/|⟨κ⟩|,   ⟨κ⟩ = Σ(κ·w)/Σw,   w = η(1−η)   (interface weight, peaks at η=0.5)
```

**Why the average, not pointwise** — pointwise `σ(R=1/κ)` is *unstable*: the diffuse-interface
curvature is noisy, and the piecewise σ(R) nonlinearity *rectifies* that noise into a net spurious
force that drifts the bubble (observed to run away to R/R₀≈1.2 in earlier diagnostics). Using one
global R per level makes σ **uniform over the shell** — which is both physically correct (the coating
tension is set by the bubble radius) and reduces the CSF force to the constant-σ form, which is
stable (linear in κ). Validated below.

## 3. Implementation (add-on; no wheel reinvented)

- **`src/Integrator/Hydro2.H`** — members `marmottant`, `marmottant_R_buckling`, `marmottant_chi`,
  `marmottant_sigma_break` (+ diagnostic `marmottant_R`).
- **`Hydro2.cpp` Parse** — `marmottant`, `marmottant.R_buckling`, `marmottant.chi`,
  `marmottant.sigma_break`. `sigma` is reused as **σ_water**.
- **`Hydro2.cpp` RHS** — after the curvature loop, a weighted reduction over the level computes
  `⟨κ⟩ → R → σ(R)` **once**; the existing CSF force just uses `sigma_eff = σ(R)` instead of `sigma`
  (one line). `marmottant=0` is byte-for-byte the old constant-σ path.

## 4. Validation (`tests/FlowMarmottant`)

**Laplace coated bubble:** a static gas bubble (R₀=0.2) with the IC pressure jump set to the Laplace
balance `ΔP = (d−1)σ(R₀)/R₀` for the Marmottant tension. Params: `χ=0.1, R_buckling=0.1` →
`σ(R₀)=0.1·(0.2²/0.1²−1)=0.3`. If R→σ(R) is right, the bubble sits at R₀ and does not drift.

| | curvature R | σ_eff | expected σ(R₀) | R(t)/R₀ drift | verdict |
|---|---|---|---|---|---|
| **2D** (`_2D`, 128²) | **0.2001** | **0.3005** | 0.300 | **0.20%** | PASS ✅ |
| **3D** (`_3D`, 64³) | _[run]_ | _[run]_ | 0.300 | _[run]_ | _[run]_ |

The 2D case nails R=R₀, σ=σ(R₀), and is **stable to 0.2% — no R/R₀→1.2 runaway**, confirming the
averaged-curvature approach fixes the pointwise-rectification instability. Checker:
`reference/check_marmottant.py` (volume-based R(t)); the solver prints `Marmottant ... R=... sigma_eff=...`.

## 5. Limitations / extensions
- Stateless σ(R) (no rupture hysteresis).
- One global R **per level** → assumes a single, near-spherical bubble. Multiple bubbles would need
  per-bubble (connected-component) curvature averaging.
- Dynamic (oscillating-bubble) validation against the Marmottant RP solution is a recommended next test.

---

## References

**Model (use ONLY these, per task):** in `flames_PAPERS/Marmottant/`
1. **P. Marmottant, S. van der Meer, M. Emmer, M. Versluis, N. de Jong, S. Hilgenfeldt, D. Lohse
   (2005).** *A model for large amplitude oscillations of coated bubbles accounting for buckling and
   rupture.* J. Acoust. Soc. Am. **118**(6), 3499-3505. doi:10.1121/1.2109427.
   — file `A_model_for_large_amplitude_oscillations_of_coated.pdf`. **σ(R), Eq. (4).**

**Radius-from-curvature (researched, as permitted — no single paper does this):**
2. **J. U. Brackbill, D. B. Kothe, C. Zemach (1992).** *A continuum method for modeling surface
   tension.* J. Comput. Phys. **100**, 335-354. doi:10.1016/0021-9991(92)90240-Y. — defines the CSF
   curvature as the divergence of the unit normal, `κ = ∇·n̂`.
   <https://www.sciencedirect.com/science/article/abs/pii/002199919290240Y>
3. *Mean curvature* (the divergence-of-unit-normal identity; mean curvature of a d-sphere = (d−1)/R).
   <https://en.wikipedia.org/wiki/Mean_curvature>
