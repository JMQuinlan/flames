# FlowAMRVerification

Regression tests for AMR coarse-fine ghost handling in `Hydro2::FillGhost4BC`.

## Tests

### `input_PressurePulseToBoundary`

Targets a specific bug pattern (fixed in the same commit as this test was
added): the `FillPatch` block at refined levels in `Hydro2::FillGhost4BC`
omitted `energy0_mf` and `energy1_mf`, leaving per-phase `E_k` coarse-fine
ghost cells unfilled. When a pressure wave reached a fine patch whose
ghost cells sat at the corner of (coarse-fine boundary) AND (physical
boundary), the BC fill operated on top of garbage `E_k` ghosts, producing
NaN.

**Setup:** sharp 50%-over-ambient Gaussian pressure pulse at the center
of a 4 cm × 4 cm air-filled domain. 3 levels of AMR refining on
`p_refinement_criterion`. Plain Neumann BCs on all four walls (the bug
is BC-agnostic — the regression should pass for the base AMReX BC
class, not just custom NSCBC).

**Expected runtime:** ~30 seconds on a modern workstation.

**Pass criterion:** the run completes to `stop_time = 3.0e-4 s` and
writes the final plotfile. NaN aborts kill the run before that.

**With the pre-fix `Hydro2.cpp`:** the run aborts with
`NaN DETECTED IN FillGhost4BC` within ~0.5–1.0 ms (about when the wave
reaches the boundary, before reaching `stop_time = 3.0e-4 s`).

**With the fix applied:** the run completes cleanly. You should see
several plotfiles (one IC + ~10 at `plot_dt = 1.0e-4`), with the
pressure wave propagating out through the boundaries.

## How to run

From the `flames2` root, after `make hydro2-2d-g++`:

```bash
./bin/hydro2-2d-g++ tests/FlowAMRVerification/input_PressurePulseToBoundary
```

To run as part of the regression suite (if you use `scripts/runtests.py`):

```bash
./scripts/runtests.py FlowAMRVerification
```

## How to verify the fix (quick check)

If you want to confirm the fix matters, comment out the two new
`FillPatch` lines for `energy0_mf` / `energy1_mf` in `Hydro2::FillGhost4BC`
and rerun. You should see an abort within ~1 ms with a NaN message
mentioning `E0` or `E1`. Restore the lines and rerun; the run should
complete cleanly.
