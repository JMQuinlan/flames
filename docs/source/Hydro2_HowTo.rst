.. _hydro2-howto:

========================================
:fas:`tasks;fa-fw` Hydro2: How-To
========================================

Short recipes for the things that come up on every Hydro2 run.

.. contents::
   :local:
   :depth: 1


Build the 2D and 3D executables
===============================

The dimension is fixed at configure time; each configuration builds into its own object directory and
its own executable, so the 2D and 3D builds live side by side.

.. code-block:: bash

   ./configure --dim=2 --get-eigen && make -j 8     # -> bin/hydro2-2d-g++
   ./configure --dim=3 --get-eigen && make -j 8     # -> bin/hydro2-3d-g++

The executable is named :code:`bin/hydro2-<dim>d-<options>-<compiler>`: :code:`--debug` adds :code:`debug`,
:code:`--omp` adds :code:`omp`, and so on.  :code:`./configure --help` lists every option.

.. code-block:: bash

   ./configure --dim=3 --omp --get-eigen && make -j 8     # -> bin/hydro2-3d-omp-g++
   OMP_NUM_THREADS=4 mpirun -np 8 --bind-to none ./bin/hydro2-3d-omp-g++ input

.. note::

   Hydro2 builds and runs in **both 2D and 3D**.  (The note in the README that it is 2D only is out of
   date.)  A few packages are narrower: the simple two-cell NSCBC is 2D only, :code:`solid.phi.ic.type = stl`
   is 3D only, and the deforming solid and ablation are single-level so far.

.. _hydro2-howto-stale:

The stale-dependency trap
-------------------------

The Makefile regenerates a dependency (:code:`.d`) file only when its own :code:`.cpp` / :code:`.cc` changes.
If you **add a new header** that an existing header includes (e.g. a new :code:`Hydro2_*.H` pulled in by
:code:`Hydro2.H`) and later edit it, :code:`src/hydro2.cc` is *not* recompiled.
The object files then disagree about the class layout and the binary dies at start-up with heap
corruption (:code:`Fatal glibc error: malloc.c ... sysmalloc`).

.. code-block:: bash

   touch src/hydro2.cc        # after editing a newly added header, before make
   make -j 8

Do it for both the 2D and the 3D build.  To check that the dependency is known:

.. code-block:: bash

   grep -c Hydro2_Solid.H obj/obj-2d-g++/hydro2.cc.d      # 0 = stale


Run a deck
==========

Decks are written to be run **from** :code:`bin/`:

.. code-block:: bash

   cd bin
   mpirun -np 8 ./hydro2-2d-g++ ../tests/FlowVortexShed/UNIT_TEST_2D_Re40 plot_file=runs/Re40/output

Anything after the deck name is a command-line override, :code:`key=value`, and wins over the deck.
Always override :code:`plot_file` so a test never writes into the source tree.

.. code-block:: bash

   mpirun -np 4 ./hydro2-2d-g++ ../tests/FlowAblation/input_1D_Stefan \
       plot_file=runs/Stefan_lam0.05/output ablation.Qstar=3.6952 stop_time=2.0

Quoting rules:

* A value with spaces (an array) needs shell quotes around the whole override:
  :code:`'amr.n_cell=100 4'`, :code:`'velocity0.ic.constant.value=1.0 0.0'`.
* An **expression** with spaces must keep its own double quotes *inside* the shell quotes,
  :code:`'key="expr with spaces"'`:

.. code-block:: bash

   mpirun -np 2 ./hydro2-2d-g++ ../tests/FlowElasticSolid/input_1D_CuAlShock \
       'eta.ic.expression.region0="1.0e-4 + (1.0 - 2.0e-4)*(x < xi)"' \
       eta.ic.expression.constant.xi=0.4 plot_file=runs/adv/output

Output
------

:code:`plot_file` is a **directory**.  It holds one :code:`NNNNNcell` plotfile per output, a
:code:`celloutput.visit` index, :code:`metadata` (every input the run actually read, with defaults filled in,
plus the git hash) and :code:`diff.patch` (the uncommitted changes of the source tree at build time).
If the directory exists it is renamed, never overwritten.

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Input
     - Default
     - Meaning
   * - :code:`plot_file`
     - :code:`output`
     - Output directory.
   * - :code:`amr.plot_dt`
     - -1
     - Simulation time between plotfiles (negative = ignored).
   * - :code:`amr.plot_int`
     - -1
     - Steps between plotfiles (negative = ignored).
   * - :code:`amr.max_plot_level`
     - -1
     - Maximum refinement level written.
   * - :code:`amr.thermo.plot_dt`, :code:`amr.thermo.plot_int`
     - -1
     - Interval of the integrated time series :code:`thermo.dat` (switches on the bubble diagnostics).
   * - :code:`plot_minimal`
     - 0
     - 1 = omit the diagnostic fields from the plotfiles.

.. tip::

   :code:`metadata` is the ground truth for *which key the parser read*.  If an input does not show up
   there under the name you typed, it was not used.

Restart
-------

.. code-block:: none

   restart = runs/Re40/output/00120cell

:code:`restart` (or :code:`restart_cell`) takes a plotfile directory.  The deck's :code:`amr.max_level` must not be
smaller than the finest level in the file.
For embedded-solid decks set :code:`solid.wall_flux` explicitly: the automatic wall selection is not
re-measured on a restart and falls back to the diffuse wall (:ref:`hydro2-wall-selection`).


Choose the time step
====================

Set :code:`dynamictimestep.on = 1` and let the solver pick the step; :code:`timestep` is then only the first
step and :code:`dynamictimestep.min` / :code:`dynamictimestep.max` bound it (both default to :code:`timestep`,
so set them).  Each step the limit is the smallest of

.. math::

   \Delta t_\text{acoustic} = C\,\frac{\Delta x}{c_\text{max} + |\mathbf{u}|_\text{max}}, \qquad
   \Delta t_\text{viscous} = C_v\,\frac{\Delta x^2}{2\,d\,\nu_\text{total}}, \qquad
   \Delta t_\text{capillary} = C_v\sqrt{\frac{\rho_\text{min}\Delta x^3}{2\pi\sigma}}, \qquad
   \Delta t_\text{force} = C_v\sqrt{\frac{\Delta x}{a_\text{max}}},

times a safety factor of 0.9, with :math:`C` = :code:`cfl`, :math:`C_v` = :code:`cfl_v` and :math:`d` the number
of dimensions.

* :code:`cfl` controls the acoustic limit.  The decks use 0.3--0.4.  With an elastic solid the wave speed
  is the longitudinal one, :math:`c_l^2 = c^2 + \tfrac43\mu/\rho`.
* :code:`cfl_v` (default = :code:`cfl`) controls everything diffusive.  :math:`\nu_\text{total}` is the largest
  :math:`\mu(\eta)/\rho` in the domain, multiplied by :math:`\max(1, \gamma/\mathrm{Pr})` with heat conduction
  on, and it includes the solid diffusivity :math:`k_s/(\rho_s c_{p,s})` (in-depth conduction) and the shell
  diffusivity :math:`\nu_s` (:ref:`hydro2-shell`).

Things that are **not** in the limit: :code:`solid.mu_factor > 1`, and a stiff explicit Brinkman penalty
(:code:`solid.implicit = 0` needs :math:`\lambda\Delta t \lesssim 2.5` on every level).

Use a Runge--Kutta integrator; without these two lines the run is Forward Euler:

.. code-block:: none

   integration.type    = RungeKutta
   integration.rk.type = 3            # SSPRK3


.. _hydro2-howto-limiter:

Pick a limiter and a Riemann solver
===================================

.. code-block:: none

   Riemann_Solver.type = hllc
   Limiter.type        = minmod

Riemann solvers
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - :code:`Riemann_Solver.type`
     - Use
   * - :code:`hllc`
     - The 6-equation HLLC of Saurel et al. (2009).  The workhorse; used by nearly every deck.
   * - :code:`hllc_elastic`
     - HLLC with the solid stress (Favrie et al. 2009).  Selected automatically with :code:`elastic.on = 1`;
       reduces to :code:`hllc` for fluids.
   * - :code:`hllc_all_mach`
     - Low-Mach (Guillard--Viozat) scaling of the pressure dissipation.
       Parameters :code:`ma_cutoff` (0.3), :code:`alpha_min` (0.0).
   * - :code:`hllc_all_mach_furfaro`
     - All-Mach variant; parameter :code:`interface_pressure_relaxation`.
   * - :code:`hlle`, :code:`hllce`
     - Alternative solvers (HLLE is the more dissipative two-wave solver).
   * - :code:`roe`
     - Roe solver; parameters :code:`entropy_fix` (0), :code:`entropy_fix_eps` (0.05).
       **This is what you get if** :code:`Riemann_Solver.type` **is left out.**
   * - :code:`hllc_oomar_jaiman`
     - Present but marked "doesn't really work" in the source.

Limiters
--------

The limiter reconstructs the primitive variables at the faces.

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - :code:`Limiter.type`
     - Order
     - Notes
   * - :code:`godunov`
     - 1
     - No reconstruction.  The default if :code:`Limiter.type` is left out.  Diffusive, but exact for
       solid|gas interface advection.
   * - :code:`minmod`
     - 2
     - Most dissipative second-order choice; the standard for embedded-solid and elastic decks.
   * - :code:`vanleer`
     - 2
     -
   * - :code:`muscl`
     - 2
     - Piecewise linear with a selectable slope: :code:`slope` = :code:`minmod` (default), :code:`mc`,
       :code:`vanleer`, :code:`superbee`.
   * - :code:`muscl2`
     - 2
     - The MUSCL scheme of Schmidmayer et al. (2020): monotonized-central slope.  No parameters.
   * - :code:`weno3`
     - 3
     - Parameters :code:`eps` (1e-6), :code:`rel_eps` (true), :code:`formulation` = :code:`jiang_shu` (default)
       or :code:`weno_z`.
   * - :code:`weno5`
     - 5
     - Same parameters.  Needs :code:`nghost >= 3` (use 4).
   * - :code:`thinc`
     - --
     - THINC sharpening of :math:`\eta` in interface cells on top of a base limiter.  Parameters
       :code:`thinc_base` (default :code:`muscl2`), :code:`thinc_beta` (1.6), :code:`thinc_eps` (1e-4).

.. warning::

   **Solver and limiter parameters live under the type name**: :code:`Limiter.thinc.thinc_base`,
   :code:`Limiter.weno3.formulation`, :code:`Limiter.muscl.slope`,
   :code:`Riemann_Solver.hllc_all_mach.ma_cutoff`.
   The header comments say "prefix :code:`Limiter.`" and one deck sets :code:`Limiter.thinc_base`; that key is
   not the one the parser reads (check :code:`metadata` of any run).

To reproduce the two-step (midpoint) time integration that Schmidmayer et al. pair with MUSCL, give the
Butcher tableau explicitly:

.. code-block:: none

   Limiter.type            = muscl2
   integration.type        = RungeKutta
   integration.rk.type     = 0
   integration.rk.weights  = 0.0 1.0
   integration.rk.nodes    = 0.0 0.5
   integration.rk.tableau  = 0.0 0.5 0.0

The alpha row: eta_consistent_advect
------------------------------------

The volume-fraction (:math:`\alpha`) row is non-conservative and is discretised separately from the
limiter:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - :code:`eta_consistent_advect`
     - :math:`\alpha` face value
   * - 0 (default)
     - First-order **donor cell**, upwinded on the contact speed.  The validated scheme for bubble
       dynamics.
   * - 1
     - Taken from the **limiter-reconstructed** state, so the :math:`\alpha` row has the same truncation
       error as the mass and energy rows.

Which one to use depends on the problem:

* **Bubble collapse / rebound: use 0.**  On outflow faces the reconstructed value adds terms that break
  the upwind maximum principle; at a near-singular collapse (gas core a few cells wide, large
  :math:`\nabla\cdot\mathbf{u}`) that decouples gas volume from gas mass and broke the uncoated collapse.
  The solver prints a warning whenever the flag is 1.
* **Metal|gas elastic cases with a higher-order limiter: use 1.**  With donor-cell :math:`\alpha` and
  limiter-reconstructed phasic energies the two do not agree at a stiff interface (copper|air:
  :math:`\pi/p \approx 3.4\times10^5`), a uniform-pressure, uniform-velocity interface is not preserved, and
  the run is unstable with every higher-order limiter.  Making the :math:`\alpha` row consistent removes the
  mismatch:

.. code-block:: none

   Limiter.type          = minmod
   eta_consistent_advect = 1

.. note::

   Validation (:code:`OPTIMIZATION_CHANGELOG.md` II.25, 2026-10-06): copper|air interface advected at
   1 km/s on 100 cells -- with the donor-cell :math:`\alpha` row and :code:`minmod` the interface pressure
   grows to 24 times ambient by 0.38 ms; with :code:`eta_consistent_advect = 1` it stays within 1.5 %
   (:code:`minmod`, :code:`vanleer`; :code:`godunov` 3 %).  :code:`weno3` still drifts: do not use it at a
   metal|gas interface.  The copper plate impact then runs with :code:`minmod` at 400 x 400 and
   1000 x 1000 (mass drift 0.006 %).  The :code:`Parse` warning ("not validated") predates this and refers
   to bubble collapse, where 0 remains the setting to use.


Set up a solid
==============

.. list-table::
   :header-rows: 1
   :widths: 24 44 32

   * - I want...
     - Minimum inputs
     - Start from
   * - a static body
     - :code:`apply_embedded_solid = 1`; :code:`solid.phi.ic`; :code:`solid.density.ic`, :code:`solid.pressure.ic`,
       :code:`solid.momentum.ic` = 0; :code:`solid.wall_flux = 1` for a thin :math:`\phi`
     - :code:`FlowVortexShed/UNIT_TEST_2D_Re40`
   * - a spinning / sliding wall
     - the same, with :code:`solid.momentum.ic` = :math:`\rho_s\mathbf{u}_s`
     - :code:`FlowRotatingCylinder/UNIT_TEST_2D_Re100`
   * - a translating body
     - the same, with :code:`solid.moving = 1` and :code:`solid.phi.ic`, :code:`solid.momentum.ic` as expressions of :code:`t`
     - :code:`FlowMixingCylinder/UNIT_TEST_2D`
   * - a body that deforms
     - :code:`elastic.on = 1`, :code:`elastic.mu<k>`, :code:`elastic.rhoref<k>`; the body is drawn with :code:`eta.ic`;
       **no** :code:`apply_embedded_solid`; :code:`amr.max_level = 0`
     - :code:`FlowElasticSolid/UNIT_TEST_2D_ShockCylinder`
   * - a body that ablates
     - a static body with :code:`solid.wall_flux = 1`, :code:`solid.visc_mirror = 1`, :code:`thermal.conduction = 1`,
       plus :code:`ablation.on = 1`, :code:`ablation.T_wall`, :code:`ablation.Qstar`, :code:`ablation.rho_s`;
       :code:`amr.max_level = 0`
     - :code:`FlowAblation/UNIT_TEST_2D_Wedge`

Details: :ref:`hydro2-solids`, :ref:`hydro2-elastic`, :ref:`hydro2-ablation`.

Rules of thumb:

* Make the :math:`\phi` width about **half a finest cell** (:code:`eps` of the tanh, or
  :code:`supersample = 8` for a bitmap) and set :code:`solid.wall_flux = 1`.  :math:`\phi` then carries the true
  surface position for the sharp wall.
* A viscous wall needs :code:`thermal.conduction = 1`, or the dissipation heat piles up at the wall.
* An isothermal wall in a stream, and a wall moving along itself, need :code:`solid.visc_mirror = 1`.
* Add :code:`solid.force_int = 1` and a :code:`refine_box` around the body if you want lift and drag.
* Use :code:`nghost = 4`.


Refine the mesh
===============

A cell is tagged when a scaled gradient exceeds its criterion,
:math:`2\,|\nabla q|\,\Delta r > \text{criterion}`, with :math:`\Delta r` the cell diagonal.

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Input
     - Default
     - Tags on the gradient of
   * - :code:`eta_refinement_criterion`
     - 0.001
     - volume fraction :math:`\eta` (the multiphase interface)
   * - :code:`phi_refinement_criterion`
     - 0.001
     - solid indicator :math:`\phi` (the solid boundary)
   * - :code:`omega_refinement_criterion`
     - 0.01
     - vorticity
   * - :code:`gradu_refinement_criterion`
     - 0.01
     - velocity (norm of the velocity gradient)
   * - :code:`p_refinement_criterion`
     - 1e-3
     - pressure
   * - :code:`rho_refinement_criterion`
     - 1e-6
     - density

The defaults are tight, and :math:`p` and :math:`\rho` are dimensional: **switch off what you do not want** by
setting the criterion to a huge number, and keep the one that follows your physics.

.. code-block:: none

   eta_refinement_criterion   = 1e10
   omega_refinement_criterion = 0.5        # follow the wake
   gradu_refinement_criterion = 1.0e8
   p_refinement_criterion     = 1e10
   rho_refinement_criterion   = 1e10
   phi_refinement_criterion   = 0.1        # and the body

A geometric box forces refinement in a region regardless of the solution:

.. code-block:: none

   refine_box.lo = -0.2 -0.3
   refine_box.hi =  1.3  0.3
   refine_box.max_level = 5      # optional: the box forces levels up to this one; above it only the criteria tag

Use the box when symmetry matters (gradient tagging plus box clustering produce asymmetric grids from a
symmetric flow) and when the force diagnostic needs the finest level to cover the whole body.

Mesh inputs that matter for Hydro2: :code:`amr.max_level`, :code:`amr.n_cell`, :code:`amr.blocking_factor`,
:code:`amr.max_grid_size`, :code:`amr.regrid_int` (default 2), :code:`amr.n_error_buf`, :code:`amr.grid_eff`.
With a small :code:`amr.blocking_factor` the executable raises :code:`amr.n_proper` for you (:ref:`hydro2-bc`).

.. note::

   The diffuse interface must stay resolved when you change the level count: :code:`epsilon` and the finest
   :math:`\Delta x` have to move together.  Dropping :code:`amr.max_level` alone can put the band below one cell.


.. _hydro2-howto-checks:

Run the check scripts
=====================

Every test folder has a :code:`reference/` directory with one analysis script per deck.
The header of each deck gives the exact two lines -- how to run it and how to check it:

.. code-block:: none

   # Run (from bin/): mpirun -np 4 ./hydro2-2d-g++ ../tests/FlowAblation/input_1D_Stefan plot_file=<dir>/output
   # Check:           python3 ../tests/FlowAblation/reference/stefan_check.py <dir>

The scripts take the **run directory** (the one that contains :code:`output/`), print the comparison with
the exact or literature value, and most write their figures under :code:`reference/Images/`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Folder
     - Scripts in :code:`reference/`
   * - :code:`tests/FlowVortexShed`
     - :code:`cylinder_re40_check.py` (Cd, Cl, L/D, separation angle), :code:`cylinder_force_history.py`
   * - :code:`tests/FlowRotatingCylinder`
     - :code:`rotating_cylinder_check.py` (reads the tabulated references in :code:`literature/`)
   * - :code:`tests/FlowMixingCylinder`
     - :code:`mixing_cylinder_check.py` (mass drift of each gas, mixing thickness)
   * - :code:`tests/FlowMovingWall`
     - :code:`stokes_check.py`
   * - :code:`tests/FlowElasticSolid`
     - :code:`elastic_waves_check.py`, :code:`cual_shock_check.py`, :code:`shock_impact_1d_check.py`,
       :code:`piston_check.py`, :code:`five_wave_check.py`, :code:`homogeneous_check.py`,
       :code:`rotating_disc_check.py`, :code:`oblique_wave_check.py`, :code:`oblique_interface_check.py`,
       :code:`line_source_check.py`, :code:`slab_vibration_check.py`, :code:`deforming_solid_2d_check.py`
   * - :code:`tests/FlowAblation`
     - :code:`stefan_check.py` (:code:`--qstar`, :code:`--twall`), :code:`solid_conduction_check.py`
       (:code:`contact` | :code:`stefan2`), :code:`gap_conduction_check.py`, :code:`inclined_stefan_check.py`,
       :code:`cylinder_check.py`, :code:`flat_plate_check.py`, :code:`blunt_body_check.py`,
       :code:`ablation_wedge_check.py`
   * - :code:`tests/FlowPhaseChange`
     - :code:`equilibrium_0d_check.py`, :code:`cavitation_tube_check.py`, :code:`cavitation_bubble_check.py`,
       :code:`saturation_table.py`
   * - :code:`tests/FlowMarmottant`
     - :code:`analyze_*.py` per deck, :code:`marmottant_rpe_km.py` (reference ODE)

When a parameter is changed on the command line, pass the same value to the checker where it has a
matching option, e.g.

.. code-block:: bash

   mpirun -np 4 ./hydro2-2d-g++ ../tests/FlowAblation/input_1D_Stefan plot_file=runs/dT1/output \
       ablation.T_wall=0.99 ablation.Qstar=0.1757
   python3 ../tests/FlowAblation/reference/stefan_check.py runs/dT1 --twall 0.99 --qstar 0.1757

.. tip::

   Gate the long runs on the short one.  Run the 1D deck of a feature and its checker first; only start
   the 2D / high-resolution case once that passes.


Troubleshooting
===============

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Look at
   * - Heap corruption / hang at start-up after a source edit
     - :ref:`hydro2-howto-stale`: :code:`touch src/hydro2.cc`.
   * - Inflow density is twice what was asked for
     - Dirichlet :code:`density.bc` without :code:`density0.bc` / :code:`density1.bc` (:ref:`hydro2-bc`).
   * - Pressure drifts / a standing wave rings in the box
     - Subsonic far field without NSCBC, or an under-damped :code:`nscbc.<face>.sigma`.
   * - Density climbs in the first solid cell at a stagnation point
     - Mass trap of the diffuse wall with a thin :math:`\phi`: :code:`solid.wall_flux = 1`.
   * - Wall temperature grows without bound
     - :code:`thermal.conduction = 1`.
   * - Wall heat flux / skin friction low, first-cell velocity too high
     - :code:`solid.visc_mirror = 1`.
   * - Lift drifts at zero angle of attack
     - Asymmetric AMR grid: refine with a symmetric :code:`refine_box` only.
   * - Coated-bubble run slows to a crawl when a level is added
     - Shell viscosity limit :math:`\Delta t\sim\Delta x^3` (:ref:`hydro2-shell`).
   * - Metal|gas interface noisy / unstable with minmod or WENO
     - :code:`eta_consistent_advect = 1`, or fall back to :code:`Limiter.type = godunov`.
   * - The run is first-order in time
     - :code:`integration.type = RungeKutta` is missing.
   * - Results differ between rank counts
     - Known for the sharp wall (:ref:`hydro2-solids`).
