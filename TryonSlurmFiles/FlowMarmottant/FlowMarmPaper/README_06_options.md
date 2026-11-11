# 06 Cattaneo on INCLINE: options 5 and 6 (2026-11-10)

Both patches change only `src/Integrator/Hydro2.{cpp,H}` (shell Gamma transport). Patches are
against commit 9087521de and were checked with `git apply --check`.

- option 5 (`06_opt5/option5.patch`): Gamma per eta level set. The Gamma row uses the raw surface
  divergence; after each step the level-set slip is applied with an implicit (backward-Euler)
  upwind shift and an exponential area stretch exp(-kappa s dt). It prints `SLIP5` lines if a level
  set slips more than 0.3 cells in one step.
- option 6 (`06_opt6/option6.patch`), SLIP OFF (changed 2026-11-10 20:20): the call to the slip step
  is commented out, so this is the old code with one change: the Gamma row uses the raw surface
  divergence div_s u in every band cell (no LiquidInterfaceRate substitution on the liquid side).
  Gamma moves with the fluid only. The conservative slip function is still compiled but never called.
  Why: locally, slip on split the 2x2 slab columns at 19.2 cells/R0 (Gamma -> 1e200) and blew up
  in gas-side trace cells; slip off stayed coherent (column spread < 1.4%).
  Binary goes to `bin_opt6noslip`, output to `output_06_Cattaneo_<HighPa|LowPa>_opt6noslip`.

Build both binaries (from a clean tree at 9087521de):

    cd /home/ttryon/flames
    git apply TryonSlurmFiles/FlowMarmottant/FlowMarmPaper/06_opt5/option5.patch
    make -j 32 bin/hydro2-3d-g++ && mkdir -p bin_opt5 && cp bin/hydro2-3d-g++ bin_opt5/
    git apply -R TryonSlurmFiles/FlowMarmottant/FlowMarmPaper/06_opt5/option5.patch
    git apply TryonSlurmFiles/FlowMarmottant/FlowMarmPaper/06_opt6/option6.patch
    make -j 32 bin/hydro2-3d-g++ && mkdir -p bin_opt6noslip && cp bin/hydro2-3d-g++ bin_opt6noslip/
    git apply -R TryonSlurmFiles/FlowMarmottant/FlowMarmPaper/06_opt6/option6.patch

Submit:

    sbatch TryonSlurmFiles/FlowMarmottant/FlowMarmPaper/06_opt5/06_Cattaneo_HighPa_opt5.sh   # and LowPa, and opt6

Each job writes to its own `output_06_Cattaneo_<HighPa|LowPa>_opt5` or `_opt6noslip`. The jobs use constant
kappa_s (`shell.kappa_s_alpha=0`); the deck's default is the Cross law.

If you already built `bin_opt6` from the earlier (slip on) option6.patch, it is NOT the slip-off binary:
reverse that patch with the copy you applied (or `git checkout src/Integrator/Hydro2.cpp src/Integrator/Hydro2.H`),
then apply this one and build `bin_opt6noslip`.
