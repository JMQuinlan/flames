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
