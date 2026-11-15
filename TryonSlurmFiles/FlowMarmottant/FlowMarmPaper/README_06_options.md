# 06 Cattaneo on INCLINE: options 5 and 6 (2026-11-10)

One binary, built from the repo as usual (`make bin/hydro2-3d-g++`). The shell transport option
is chosen in the deck with `shell.slip`:

| deck | slurm | `shell.slip` | what it is |
|---|---|---|---|
| `input_06_Cattaneo_<HighPa,LowPa>_opt5` | `06_Cattaneo_<HighPa,LowPa>_opt5.sh` | 5 | Gamma follows the eta level sets (implicit upwind slip + exp(-kappa s dt) area stretch); prints `SLIP5` lines on large slips |
| `input_06_Cattaneo_<HighPa,LowPa>_opt6` | `06_Cattaneo_<HighPa,LowPa>_opt6.sh` | 0 | option 6 with slip OFF: Gamma moves with the fluid only, Gamma row with the raw surface divergence div_s u |

Both decks are the base 06 deck with constant kappa_s (`shell.kappa_s_alpha = 0`, the paper's
model), their own `plot_file` (`output_06_Cattaneo_<HighPa,LowPa>_opt<5,6>`), and a faster AMR
layout: `amr.max_grid_size = 16`, `amr.n_error_buf = 4` (base deck: 32 / 16). With 32 / 16 every level
had only 8 boxes, so at most 8 ranks worked; n_error_buf 4 cuts finest cells 2.4x and ran 1.9x faster locally.
The slurm scripts use 32 tasks; `06_Cattaneo_opt56_all.sh` runs all four on one 128-core node.

`shell.slip` defaults to 0, so the base decks `input_06_Cattaneo_<HighPa,LowPa>` now also run
with slip off. `shell.slip = 6` (conservative slip) still exists but is broken locally: Gamma
blows up in gas-side band cells.

The folders `06_opt5/` and `06_opt6/` (patch files + scripts for separate binaries) are obsolete:
the code they patch in is already in the repo, so the patches do not apply.

## opt5_10_8 (2026-11-11): option 5 + 10 + relax.opt8 = 1
shell.slip = 5, face.two_pressure = 1, relax.opt8 = 1. Only combination to reach 6 us at 160 kPa / 19.2 cells/R0 in the
RADIAL slab: peaks vs model +1.2/+1.2/-0.3/-0.4 %, troughs -1.8/-5.9/-10.3/-11.6 % (old code +2.1/+4.6/+8.6/+15.9 %).
Slurm: 06_Cattaneo_{HighPa,LowPa}_opt5_10_8.sh (32 tasks). Needs hydro2 from be9e72cd6 or later.

## opt1 (2026-11-14): gooderrun + spherical NSCBC + Option 1 viscous relaxation jump
Decks input_06_Cattaneo_{HighPa,LowPa}_opt1, slurm 06_Cattaneo_{HighPa,LowPa}_opt1.sh (32 tasks). Flags on top of the base deck:
shell.kappa_s_alpha 0, shell.slip 12, face.two_pressure 1, relax.opt8 1, relax.trace_pT 3, thermal.alpha_min 1e-4,
nscbc.spherical 1, thermal.dt_limit 1, visc.relax_cell_local 0; plot_dt 2.5e-8, max_grid_size 16, n_error_buf 4.
Option 1: the relaxation's viscous jump obeys eta J_R = int J_F d eta, INTEGRATED ALONG THE NORMAL (grad eta) inside the
band (recovers the full shell viscosity: free-decay delivered fraction 1.03 vs 0.83 per-cell). visc.relax_cell_local = 1
restores the old per-cell jump. thermal.dt_limit fixes the fine-grid NaN at maximum expansion (explicit conduction).
