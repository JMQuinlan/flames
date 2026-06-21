# AMReX patches (must be re-applied if `ext/AMReX-Codes/amrex` is re-cloned)

## eb_custom_fab_fillpatch.patch
**Why:** With `USE_EB` (enabled by `./configure --eb` for 3D STL geometry),
AMReX's `InterpFromCoarseLevel` builds an `EBFArrayBoxFactory` for the coarse
patch, which only works for `FArrayBox`/`MultiFab`. Alamo's node tensor fields
are `FabArray<BaseFab<Eigen::Matrix>>`, so that branch fails to COMPILE (the
solid-mechanics integrators: mechanics, thermoelastic, etc.) -- even though it
is `if (index_space)`-gated and alamo never builds an EB index space, so it is
dead at runtime.

**Fix:** guard the EB-factory branch with `if constexpr (std::is_same_v<FAB,
amrex::FArrayBox>)` so it compiles out for non-FArrayBox fields.

**Apply:**  `git -C ext/AMReX-Codes/amrex apply ../../scripts/amrex_patches/eb_custom_fab_fillpatch.patch`
(then rebuild). Editing the AMReX source makes `git describe` report `-dirty`,
so the install dir becomes `<dim>-g++-<ver>-dirty` -- harmless, just a name.
