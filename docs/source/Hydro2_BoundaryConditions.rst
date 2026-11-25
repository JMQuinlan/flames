.. _hydro2-bc:

=====================================================
:fas:`border-all;fa-fw` Hydro2: Boundary Conditions
=====================================================

Domain boundary conditions are given **per field and per face**.  Every conserved field of the
6-equation model has its own :code:`<field>.bc` block, read by :code:`BC::Expression`
(:code:`src/BC/Expression.H`); non-reflecting faces are handed to the characteristic boundary condition
(:code:`BC::NSCBC` / :code:`BC::NSCBC4`) on top of that.
Walls *inside* the domain are not boundary conditions in this sense -- they are embedded solids, see
:ref:`hydro2-solids`.

.. contents::
   :local:
   :depth: 2


Syntax
======

.. code-block:: none

   <field>.bc.type.<face> = <type> [<type> ...]
   <field>.bc.val.<face>  = <value or expression> [...]
   <field>.bc.constant.<name> = <number>

* :code:`<face>` is :code:`xlo`, :code:`xhi`, :code:`ylo`, :code:`yhi` (and :code:`zlo`, :code:`zhi` in 3D).
  The :code:`z` entries are read and ignored by a 2D build, so one deck can serve both.
* A scalar field takes one type per face.  :code:`momentum` takes one **per component**
  (two in 2D, three in 3D); a single entry is applied to all components.
* :code:`val` is an expression of :code:`x`, :code:`y`, :code:`z`, :code:`t`.  Named constants are declared with
  :code:`<field>.bc.constant.<name>` and belong to that field's block only.
* A face with no :code:`val` uses :code:`0.0` for every component.

.. warning::

   **The default type is** :code:`dirichlet` **with value 0.**  A face you forget is not an outflow -- it is
   a Dirichlet zero.  Give every face of :code:`density`, :code:`momentum` and :code:`energy` explicitly.


Fields
======

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Block
     - Applied to
     - If the block is absent
   * - :code:`density.bc`
     - mixture density; also the partial densities unless overridden
     - (give it)
   * - :code:`density0.bc`, :code:`density1.bc`
     - partial densities :math:`(\alpha\rho)_0`, :math:`(\alpha\rho)_1`
     - the shared :code:`density.bc`
   * - :code:`momentum.bc`
     - mixture momentum :math:`\rho\mathbf{u}` (**momentum, not velocity**)
     - (give it)
   * - :code:`energy.bc`
     - :math:`\rho E` and the phasic internal energies :math:`E_0`, :math:`E_1`
     - (give it)
   * - :code:`eta.bc`
     - volume fraction :math:`\eta`; shared by the advected shell density and the vapour field
     - zero-gradient on every face
   * - :code:`pressure.bc`
     - pressure, only with :code:`bc.primitive = 1`
     - not used
   * - :code:`solid.phi.bc`
     - embedded-solid indicator :math:`\phi`
     - zero-gradient on every face

The optional blocks (:code:`density0.bc`, :code:`density1.bc`, :code:`eta.bc`, :code:`solid.phi.bc`) are
detected by the presence of their :code:`type.xlo` **or** :code:`type.ylo` key.

Per-phase densities at a Dirichlet face
---------------------------------------

The partial densities are filled directly from their own BC, and the mixture ghost is their sum.
A Dirichlet :code:`density.bc` value is therefore written into **both** phases: 2x the intended density for a
50/50 mixture, or the whole liquid density injected into the gas phase for a pure-phase far field.
Declare :code:`density0.bc` / :code:`density1.bc` so each phase gets its own :math:`(\alpha\rho)_k`.
The solver warns at :code:`Parse` when a face is :code:`dirichlet` and they are missing.
For :code:`neumann` / reflective / :code:`periodic` faces the shared block is exact.

Energy at a Dirichlet face
--------------------------

:code:`energy.bc` values are applied to the **phasic internal energies**
:math:`E_k = \alpha_k (p + \gamma_k\pi_k)/(\gamma_k - 1)`, not to the mixture total energy;
:math:`\rho E` in the ghost cells is rebuilt as :math:`E_0 + E_1 + \tfrac12\rho|\mathbf{u}|^2`.
With one value for both phases this is exact only when the two are equal (e.g. :math:`\eta = 0.5` and
identical EOS, as the single-fluid decks do).


Boundary types
==============

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Type
     - Ghost-cell fill
   * - :code:`neumann`
     - Zero-gradient (extrapolation) when :code:`val` = 0: ghost = nearest interior cell.
       With a value, ghost = interior - :code:`val` x :math:`\Delta x` (a prescribed gradient).
   * - :code:`dirichlet`
     - Ghost = :code:`val(x, y, z, t)`.  Aliases: :code:`EXT_DIR`, :code:`Inflow`.
   * - :code:`periodic`
     - Periodic copy.  The direction must also be periodic in :code:`geometry.is_periodic`, and every field
       must be :code:`periodic` on both faces of it.
   * - :code:`REFLECT_EVEN`
     - Mirror across the face: ghost = mirrored interior cell.  Alias :code:`Symmetry`.
   * - :code:`REFLECT_ODD`
     - Mirror with a sign change: ghost = - mirrored interior cell.
   * - :code:`nscbc_inflow`, :code:`nscbc_outflow`
     - Characteristic (non-reflecting) face; the ghost cells are written by NSCBC, see below.

Ghost cells outside two or more faces (edges, corners) are resolved in one go when every face involved
is of mirror or zero-gradient type, so the mirror composition is exact and order-independent.

.. note::

   Type strings are **case sensitive**.  The mirrors are upper case (:code:`REFLECT_ODD`,
   :code:`REFLECT_EVEN`); :code:`neumann`, :code:`dirichlet`, :code:`periodic` are lower case (a leading
   capital is also accepted).  Other names the parser knows (:code:`FOEXTRAP`, :code:`HOEXTRAP`,
   :code:`Outflow`, lower-case :code:`reflect_odd`, :code:`SlipWall`, :code:`NoSlipWall`, ...) have no
   dedicated ghost fill in :code:`BC::Expression` -- :code:`SlipWall` and :code:`NoSlipWall` are plain
   Dirichlet aliases there, not wall models -- so do not use them.


Recipes
=======

Outflow / open boundary (zero-gradient)
---------------------------------------

.. code-block:: none

   density.bc.type.xhi  = neumann
   momentum.bc.type.xhi = neumann neumann
   energy.bc.type.xhi   = neumann

Fine for supersonic outflow.  For a subsonic far field nothing holds the pressure and start-up acoustics
are trapped; use NSCBC there.

Supersonic inflow (Dirichlet)
-----------------------------

From the Mach 3 wedge deck (:math:`\rho = 1`, :math:`p = 0.714286`, :math:`U = 3`, :math:`\eta = 0.5`,
:math:`\gamma = 1.4`):

.. code-block:: none

   eta.bc.type.xlo      = dirichlet
   eta.bc.val.xlo       = 0.5
   density.bc.type.xlo  = dirichlet
   density.bc.val.xlo   = 1.0
   density0.bc.type.xlo = dirichlet
   density0.bc.val.xlo  = 0.5          # (alpha rho)_0
   density1.bc.type.xlo = dirichlet
   density1.bc.val.xlo  = 0.5          # (alpha rho)_1
   momentum.bc.type.xlo = dirichlet dirichlet
   momentum.bc.val.xlo  = 3.0 0.0      # rho u, rho v
   energy.bc.type.xlo   = dirichlet
   energy.bc.val.xlo    = 0.892857     # per-phase E0 = E1 = alpha p/(gamma-1); NOT the mixture total E

Remember that the optional blocks need *all* their faces once they are declared.

Symmetry plane / slip wall (reflective)
---------------------------------------

The normal momentum is odd, everything else even.  For the :math:`x`, :math:`y`, :math:`z` low faces of a
3D octant:

.. code-block:: none

   eta.bc.type.xlo      = REFLECT_EVEN
   density.bc.type.xlo  = REFLECT_EVEN
   energy.bc.type.xlo   = REFLECT_EVEN
   momentum.bc.type.xlo = REFLECT_ODD  REFLECT_EVEN REFLECT_EVEN
   momentum.bc.type.ylo = REFLECT_EVEN REFLECT_ODD  REFLECT_EVEN
   momentum.bc.type.zlo = REFLECT_EVEN REFLECT_EVEN REFLECT_ODD

A face whose **normal-momentum** type is :code:`REFLECT_ODD` is detected as a symmetry face
(console: *"symmetry face detected ... (exact u*=0 flux enforced)"*) and gets three extra treatments,
all on by default:

.. list-table::
   :header-rows: 1
   :widths: 24 10 66

   * - Input
     - Default
     - Meaning
   * - (automatic)
     -
     - The advective fluxes on the face are enforced to the exact :math:`u^* = 0` solution.
   * - :code:`sym_face_central`
     - 1
     - Central (not one-sided) Hessian stencils for :math:`\eta` and :math:`\rho` in the first row next to a
       symmetry face, so that row is consistent with its mirror image.
   * - :code:`omega_sym_mirror`
     - 1
     - The capillary / shell stress tensor in the ghost layer of a symmetry face is set to the mirror of the
       first valid cell.

No-slip wall at the domain edge
-------------------------------

.. code-block:: none

   density.bc.type.ylo  = neumann
   energy.bc.type.ylo   = neumann
   momentum.bc.type.ylo = dirichlet dirichlet
   momentum.bc.val.ylo  = 0.0 0.0

A wall moving along itself is the same with a non-zero tangential *momentum* value.

Periodic
--------

.. code-block:: none

   geometry.is_periodic = 1 0 0
   eta.bc.type.xlo      = periodic
   eta.bc.type.xhi      = periodic
   density.bc.type.xlo  = periodic
   density.bc.type.xhi  = periodic
   momentum.bc.type.xlo = periodic periodic
   momentum.bc.type.xhi = periodic periodic
   energy.bc.type.xlo   = periodic
   energy.bc.type.xhi   = periodic

Time-dependent (expression) boundary
------------------------------------

Any :code:`val` can be an expression of :code:`t`.  An acoustic drive on the low-:math:`x` face:

.. code-block:: none

   density1.bc.constant.rho_base = 1.0
   density1.bc.constant.amp      = 0.01
   density1.bc.constant.c0sq     = 1.4
   density1.bc.constant.omega    = 31.415926536
   density1.bc.type.xlo = dirichlet
   density1.bc.val.xlo  = "rho_base + (amp/c0sq)*sin(omega*t)"

Prescribing pressure instead of energy
--------------------------------------

With :code:`bc.primitive = 1` (default 0) the boundary state is given as density, momentum, **pressure**
and :math:`\eta`: the pressure ghost is filled from :code:`pressure.bc` and the phasic internal energies are
rebuilt from :math:`(\alpha, p)` through the EOS.  :code:`energy.bc` is still read (it carries the
AMR fill of the energy fields).  :code:`pressure.bc` is only read when no face is of an :code:`nscbc_*` type.

.. code-block:: none

   bc.primitive = 1
   pressure.bc.constant.p_base = 1.0
   pressure.bc.constant.amp    = 0.01
   pressure.bc.constant.omega  = 31.415926536
   pressure.bc.type.xlo = dirichlet
   pressure.bc.val.xlo  = "p_base + amp*sin(omega*t)"
   pressure.bc.type.xhi = neumann


Characteristic boundaries (NSCBC)
=================================

NSCBC is the ghost-cell Navier--Stokes characteristic boundary condition of Motheau et al. (2017): the
ghost cells are filled from a characteristic wave decomposition so that outgoing acoustics leave the
domain instead of ringing in it.

It is switched on by the face types.  If **any** face of :code:`density.bc` is of an :code:`nscbc_*` type the
run uses NSCBC:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - :code:`nghost`
     - Variant
   * - 2
     - :code:`BC::NSCBC`, the simple two-cell variant.  **2D only** (aborts in 3D).
   * - 4
     - :code:`BC::NSCBC4`, the four-cell variant.  2D and 3D (faces, edges and corners).  Use this.
   * - other
     - aborts: *"NSCBC requires nghost = 2 or 4"*.

A face needs two things: the field types, and a matching :code:`nscbc.<face>` block.

.. code-block:: none

   nghost = 4

   density.bc.type.xlo  = nscbc_inflow
   momentum.bc.type.xlo = nscbc_inflow nscbc_inflow
   energy.bc.type.xlo   = nscbc_inflow
   density.bc.type.xhi  = nscbc_outflow
   momentum.bc.type.xhi = nscbc_outflow nscbc_outflow
   energy.bc.type.xhi   = nscbc_outflow
   density0.bc.type.xlo = neumann        # per-phase blocks: neumann on NSCBC faces
   density0.bc.type.xhi = neumann
   density1.bc.type.xlo = neumann
   density1.bc.type.xhi = neumann

   nscbc.xlo.type     = inflow
   nscbc.xlo.target_u = 1.0
   nscbc.xlo.target_v = 0.0
   nscbc.xlo.target_T = 6.2188658e-02    # T = p/(rho cv (gamma-1))
   nscbc.xhi.type     = outflow
   nscbc.xhi.target_p = 1785.71
   nscbc.xhi.sigma    = 0.5
   nscbc.xhi.beta     = 0.5

The subsonic inflow fixes :math:`u`, :math:`v`, :math:`T`; the outflow relaxes toward the target pressure.
A face left without an :code:`nscbc.<face>.type` is owned by its field BCs -- this is how a symmetry plane
(:code:`REFLECT_*`) and NSCBC faces coexist in one deck: the :code:`BC::Expression` object honours the
reflective faces and only clamp-fills the :code:`nscbc_*` ones, which NSCBC then overwrites.

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Input (per :code:`<face>`)
     - Default
     - Meaning
   * - :code:`nscbc.<face>.type`
     - none
     - :code:`inflow` (subsonic, relaxed to a target state) or :code:`outflow` (subsonic, pressure relaxation).
   * - :code:`target_u`, :code:`target_v`, :code:`target_w`
     - 0
     - Target velocity.  On :math:`x` and :math:`y` faces NSCBC4 reads :code:`target_u` as the :math:`x` velocity
       and :code:`target_v` as the :math:`y` velocity (so the normal target on a :math:`y` face is
       :code:`target_v`); on :math:`z` faces it reads :code:`target_u` as the **normal** one.  Check the
       convention before relying on a :math:`z`-face inflow.
   * - :code:`target_T`
     - 300
     - Target temperature (inflow).
   * - :code:`target_p`
     - 101325
     - Target pressure (outflow).
   * - :code:`relax_u`, :code:`relax_v`, :code:`relax_w`, :code:`relax_T`
     - 0.3
     - Inflow relaxation coefficients (0.2--0.5 typical).
   * - :code:`beta`
     - 0.6
     - Transverse-term weight (0--1).
   * - :code:`sigma`
     - 0.25
     - Outflow pressure relaxation coefficient, non-dimensional: the rate is
       :math:`K = \sigma(1 - M^2)\,a/L_\text{ref}`.  Below 0.2 the outflow is under-damped (warning); 0.25--0.6
       recommended.
   * - :code:`L_ref`
     - domain/20 (NSCBC4); 1.0 (2-cell)
     - Reference length in :math:`K`.  NSCBC4 defaults it to the largest domain extent / 20 when not given.
   * - :code:`drive_amp`, :code:`drive_omega`, :code:`drive_phase`
     - none
     - Time-dependent pressure drive through the incoming characteristic,
       :math:`p(t) = p_\text{target} + \sum_k A_k\sin(\omega_k t + \varphi_k)`.
       NSCBC4 takes arrays (a Fourier series, up to 8 terms); amplitudes and frequencies must have the same
       length, phases are optional.

Global: :code:`nscbc.small` (default 1e-10) and, for NSCBC4, :code:`nscbc.ghost_mode` (default 3 =
first-order LODI with linear gradient-extrapolated ghosts; 0 = legacy layered path, known to amplify
grid-scale waves; 1 and 2 are diagnostics).

.. note::

   Driving a bubble acoustically is done with :code:`drive_amp` / :code:`drive_omega` on an NSCBC outflow
   face, not with a Dirichlet pressure: outgoing waves still leave, so the face drives *without* acting as
   a resonator wall.


Refinement and ghost cells
==========================

A fine level's ghost region is interpolated from its parent, which needs the parent's valid data to
cover the fine box grown by :code:`nghost`.  With :code:`amr.blocking_factor < 2*nghost` and the default
proper nesting that is not guaranteed.  The :code:`hydro2` executable therefore raises :code:`amr.n_proper`
automatically before the mesh is built (console: *"amr.n_proper auto-set to ..."*); an explicit
:code:`amr.n_proper` in the deck wins, and a warning is printed if it is too small.


Known limitations
=================

* :code:`nscbc.<face>.type` also accepts :code:`slipwall` and :code:`noslipwall`, but the NSCBC4 ghost fill
  only has inflow and outflow branches; the wall types are not exercised by any deck.  Use a reflective
  face or a Dirichlet momentum instead.
* The two-cell NSCBC is 2D only.
* The optional blocks are only detected through their :code:`xlo` / :code:`ylo` keys.
* With :code:`dynamictimestep.on` the time label handed to a time-dependent BC is ahead by about one step.
