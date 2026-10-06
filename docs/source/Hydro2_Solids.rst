.. _hydro2-solids:

================================================
:fas:`cube;fa-fw` Hydro2: Non-deforming Solids
================================================

A non-deforming (rigid) body is embedded in the 6-equation flow through a second indicator field,
:math:`\phi`, that is independent of the fluid volume fraction :math:`\eta`:

.. math::

   \phi = 1 \;\;\text{(fluid)}, \qquad \phi = 0 \;\;\text{(solid)}.

The multiphase interface (:math:`\eta`) stays diffuse; only the treatment of the **solid** changes
between the two wall models below.
The body can be static, move with a prescribed rigid motion (:ref:`hydro2-moving-solid`),
or recede by ablation (:ref:`hydro2-ablation`).
A solid that *deforms* is a different model altogether, see :ref:`hydro2-elastic`.

Everything on this page is switched on by

.. code-block:: none

   apply_embedded_solid = 1

and is controlled by the :code:`solid.*` inputs (class :code:`EmbeddedSolid`,
:code:`src/Integrator/EmbeddedSolid.H`).

.. contents::
   :local:
   :depth: 2


Geometry and solid state
========================

The indicator
-------------

:code:`solid.phi.ic` sets :math:`\phi`.  Accepted types are :code:`constant`, :code:`expression`,
:code:`bmp`, :code:`png` and, in 3D builds only, :code:`stl`.
A cylinder of radius :code:`R0` at the origin is

.. code-block:: none

   solid.phi.ic.type = expression
   solid.phi.ic.expression.constant.R0  = 0.5
   solid.phi.ic.expression.constant.eps = 0.013
   solid.phi.ic.expression.region0 = "0.5*(1.0 + tanh((sqrt(x*x + y*y) - R0)/eps))"

and a step wall is simply :code:`"1.0*(y > 0.0)"`.
The width :code:`eps` decides which wall the solver will use (see :ref:`hydro2-wall-selection`), and
when it is about half a finest cell :math:`\phi` carries the sub-cell position of the true surface,
:math:`\phi = 0.5`, which the sharp wall uses.
For bitmaps use :code:`solid.phi.ic.bmp.supersample = 8` so that :math:`\phi` is a volume fraction
rather than a 0/1 mask.

The target state
----------------

Solid cells are held at a prescribed state.  The input is **single-phase**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Input
     - Meaning
   * - :code:`solid.density.ic`
     - Total density :math:`\rho_s` in the solid.
   * - :code:`solid.pressure.ic`
     - Pressure :math:`p_s` in the solid.
   * - :code:`solid.momentum.ic`
     - Momentum :math:`\mathbf{M}_s = \rho_s \mathbf{u}_s` in the solid.  Zero for a static wall; this is
       how the **wall velocity** is given.

Because the 6-equation state is two-phase, :code:`InitEmbeddedSolidTarget` splits this state over both
phase slots by the local :math:`\eta` with the stiffened-gas EOS,

.. math::

   (\alpha\rho)_0 = \eta\,\rho_s, \quad (\alpha\rho)_1 = (1-\eta)\,\rho_s, \qquad
   E_0 = \eta\,\frac{p_s + \gamma_0\pi_0}{\gamma_0 - 1}, \quad
   E_1 = (1-\eta)\,\frac{p_s + \gamma_1\pi_1}{\gamma_1 - 1}.

Set :math:`\rho_s`, :math:`p_s` to the free-stream values unless there is a reason not to.

The boundary condition of :math:`\phi` at the domain edge defaults to zero-gradient
(:code:`solid.phi.bc` overrides it, see :ref:`hydro2-bc`).


Wall models
===========

Diffuse wall (Brinkman penalization)
------------------------------------

The default treatment is the momentum-only diffuse-domain penalization of Yang (2023):
every face, including fluid|solid ones, gets the ordinary Riemann flux and only momentum is
penalized toward the solid's,

.. math::

   \frac{\partial (\rho\mathbf{u})}{\partial t} = \dots - \lambda\,(1-\phi)\,\big(\rho\mathbf{u} - \mathbf{M}_s\big),

with :math:`\lambda` = :code:`solid.brinkman`.  Mass and energy are left to evolve.
With :code:`solid.slip = 1` only the wall-normal momentum is penalized (no-penetration, slip);
with :code:`solid.slip = 0` all components are (no-slip).

With :code:`solid.implicit = 1` (default) the penalty is not an explicit RHS source but the **exact
exponential** update, applied once per level step after the Runge--Kutta advance,

.. math::

   \rho\mathbf{u} \;\leftarrow\; \rho\mathbf{u}_s + \big(\rho\mathbf{u} - \rho\mathbf{u}_s\big)\,
   e^{-\lambda (1-\phi)\Delta t}, \qquad \mathbf{u}_s = \mathbf{M}_s/\rho_s ,

and the solid's work :math:`\mathbf{u}_s\cdot\Delta(\rho\mathbf{u})` is added to :math:`\rho E` (zero for
a static body).
The explicit source is only stable for :math:`\lambda\Delta t \lesssim 2.5` on *every* level, which
caps :math:`\lambda` at :math:`O(1/\Delta t_\text{coarse})` -- far too weak for a thin wall.
The implicit update is unconditionally stable, so :math:`\lambda` can be :math:`10^4`--:math:`10^8`.

The diffuse wall is the right choice when :math:`\phi` is resolved over several cells.

Sharp wall (mirrored wall flux)
-------------------------------

With a sharp :math:`\phi` (a step) and a stiff penalty the full-flux wall is a **mass trap**: at a
stagnation point the fluid|solid face carries mass *into* the first solid cell, the penalty zeroes its
momentum, and nothing carries it out.  The sharp wall, :code:`solid.wall_flux = 1`, fixes this:

* A face whose two cells lie on opposite sides of :math:`\phi = 0.5` is solved as a **wall**: the fluid
  cell against its mirror state (normal velocity reflected about the solid's; tangential velocity too
  when :code:`solid.slip = 0`).  The contact speed is that of the wall, so no mass or energy crosses and
  only pressure acts.
* Faces with both cells in the solid carry zero flux.
* The limiter stencils next to the wall see mirrored ghost states instead of the solid cells' own
  (ghost-mirror reconstruction).  Without this a second-order limiter + AMR + solid is unstable.
* Solid cells are **inert**: their RHS is zeroed, their phasic energies are held at the target, and the
  implicit step projects them onto the target exactly, so the wall no longer depends on the value of
  :code:`solid.brinkman`.  The momentum they *would* have received is summed -- that is the force on the
  body (see :ref:`hydro2-forces`).

Where the ghost state sits is controlled by a second group of flags.  With the defaults the ghost is
built about the **true surface** :math:`\phi = 0.5` rather than the staircase face:
for a no-slip wall (:code:`solid.wall_image_noslip = 1`)

.. math::

   \mathbf{u}_g - \mathbf{u}_s = -r\,(\mathbf{u}_f - \mathbf{u}_s), \qquad
   r = \frac{0.5 - \phi_s}{\phi_f - 0.5},

which is exactly the plain mirror (:math:`r = 1`) for a 0/1 :math:`\phi`; for a slip wall
(:code:`solid.wall_normal = 1`, :code:`solid.wall_image = 1`) the ghost is reflected about the true
surface normal :math:`\mathbf{n} = \nabla\phi/|\nabla\phi|` instead of the face normal, so the step
risers of an inclined wall do not stagnate the along-surface flow.

.. _hydro2-wall-selection:

Which wall am I getting?
------------------------

When :code:`solid.wall_flux` is **not** given, the wall is chosen automatically.
At initialization the thickness of the :math:`\phi` interface is measured on each level as
:math:`1/\max|\phi_{i+1}-\phi_i|` (a step is about 1 cell, a tanh profile about :math:`2\,\text{eps}/\Delta x`),
and before the first step the finest measured level decides:

* thinner than :code:`solid.wall_flux_cells` (default 2) cells: the sharp wall, with the console message
  *"the solid is too thin to be resolved as a diffuse wall -- switching to the MIRRORED sharp-wall flux"*;
* otherwise the diffuse wall is kept (also reported).

An explicit :code:`solid.wall_flux = 0` or :code:`1` always wins.

.. warning::

   On a **restart** the thickness is not measured and the automatic choice falls back to the diffuse
   wall with a warning.  Set :code:`solid.wall_flux` explicitly in any deck that will be restarted.


Solid inputs
============

Core
----

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Input
     - Default
     - Meaning
   * - :code:`apply_embedded_solid`
     - 0
     - 1 = the domain has an embedded solid.
   * - :code:`solid.brinkman`
     - 1e6
     - Penalty rate :math:`\lambda`.  :code:`<= 0` selects the old automatic momentum-projection wall.
   * - :code:`solid.implicit`
     - 1
     - 1 = exact implicit (exponential) Brinkman update after each level step; 0 = old explicit RHS source.
   * - :code:`solid.slip`
     - 0
     - 1 = slip wall (only wall-normal momentum penalized / mirrored); 0 = no-slip.
   * - :code:`solid.wall_flux`
     - auto
     - 1 = sharp wall (mirrored Riemann flux, inert solid cells); 0 = diffuse (full-flux) wall.
       Not given = automatic selection.  Any other value aborts.
   * - :code:`solid.wall_flux_cells`
     - 2.0
     - Threshold of the automatic selection: sharp wall if the :math:`\phi` interface is thinner than this many cells.
   * - :code:`solid.mu_factor`
     - 1.0
     - Solid viscosity = :code:`mu_factor` x fluid viscosity, blended by :math:`(1-\phi)` into the Cauchy stress.
       1 = off.  Must be >= 1.  Only acts where the fluid is viscous.
   * - :code:`phi_refinement_criterion`
     - 0.001
     - AMR: tag where :math:`2|\nabla\phi|\,\Delta r` exceeds this (refines the solid boundary).

Sharp-wall geometry (only read when :code:`solid.wall_flux = 1`)
-----------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Input
     - Default
     - Meaning
   * - :code:`solid.visc_mirror`
     - 0
     - Viscous stencil at the wall.  0 = solid cells' own velocity one cell away (no-slip plane half a cell
       *inside* the solid); 1 = mirrored velocity, no-slip plane **on the fluid|solid face**;
       2 = ghost at the true surface :math:`\phi = 0.5` (ratio :math:`r`, capped at 4) -- gate failed, not adopted.
   * - :code:`solid.wall_image_noslip`
     - 1
     - No-slip walls: ghost velocity :math:`-r(\mathbf{u}-\mathbf{u}_s)`, linear to zero at :math:`\phi = 0.5`.
       Identical to the plain mirror for a 0/1 :math:`\phi`.
   * - :code:`solid.wall_normal`
     - 1
     - Slip walls: reflect the ghost about the true surface normal (ghost-cell immersed boundary); 0 = old
       face-normal mirror.  No-slip walls are unaffected.  Wants :code:`nghost >= 4`.
   * - :code:`solid.wall_image`
     - 1
     - Slip walls with :code:`wall_normal`: place the slip plane at :math:`\phi = 0.5` instead of the staircase face.
   * - :code:`solid.wall_thin_cells`
     - 2
     - :code:`wall_normal`: where the solid is at most this many cells thick along the face axis (sharp trailing
       edges, wedge tips) use the face normal -- the smoothed normal is unreliable there.  0 = never.
   * - :code:`solid.wall_redistribute`
     - 0
     - Slip + :code:`wall_normal`: hand mass/energy that crosses riser faces into solid cells back to the fluid
       neighbours.  Off: it breaks the Kutta condition at thin trailing edges.
   * - :code:`solid.wall_vel_surface`
     - 1
     - Wall velocity evaluated at the true surface point between the fluid and solid cell (exact for rigid
       rotation); 0 = the solid cell's own.  Only matters when :math:`\mathbf{u}_s` varies in space.

Motion, heat and diagnostics
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Input
     - Default
     - Meaning
   * - :code:`solid.moving`
     - 0
     - 1 = the geometry itself moves: :code:`solid.phi.ic` and the solid target are re-evaluated every step.
   * - :code:`solid.T_wall`
     - -1
     - > 0: isothermal wall at this temperature for the gas heat conduction (:code:`thermal.conduction = 1`),
       surface at :math:`\phi = 0.5`.  <= 0: adiabatic wall.
   * - :code:`solid.force_int`
     - 0
     - N > 0: every N finest-level steps append the force on the body and the probe pressures to
       :code:`<plot_file>_forces.dat`.
   * - :code:`solid.probe.x`, :code:`solid.probe.y`, :code:`solid.probe.z`
     - empty
     - Point pressure probes written to the force file (nearest cell on the finest level).  Same length;
       :code:`z` is read only in 3D, where it is required.

Recommended sharp-body block
----------------------------

This is what the viscous airfoil decks use.  The first lines are defaults and are listed for clarity.

.. code-block:: none

   apply_embedded_solid = 1
   solid.brinkman       = 1.0e6
   solid.implicit       = 1
   solid.wall_flux      = 1
   solid.slip           = 0
   thermal.conduction   = 1
   thermal.Pr           = 0.72
   solid.force_int      = 1
   nghost               = 4
   integration.type     = RungeKutta
   integration.rk.type  = 3

.. note::

   Without :code:`integration.type = RungeKutta` the base integrator uses **Forward Euler**.

Heat conduction at the wall
---------------------------

Without heat conduction a viscous no-slip wall has nowhere to put its dissipation heat (effectively
:math:`\mathrm{Pr}\to\infty`) and the wall temperature grows without bound.
:code:`thermal.conduction = 1` adds :math:`\nabla\cdot(k\nabla T)` with :math:`k = \mu(\eta)c_p(\eta)/\mathrm{Pr}`
to :math:`\rho E`, split to the phasic energies by volume fraction.
Faces that touch the solid are adiabatic unless :code:`solid.T_wall > 0`.

.. warning::

   With an **isothermal wall in a stream** set :code:`solid.visc_mirror = 1`.
   With the default 0 the no-slip plane sits :math:`\Delta x/2` inside the solid while the thermal wall is
   on the face: flat-plate wall heat flux -7 to -20 %, first-cell velocity twice too high.

.. _hydro2-forces:

Force and probe history
-----------------------

:code:`solid.force_int = N` writes one row per N finest-level steps to :code:`<plot_file>_forces.dat`:

.. code-block:: none

   # step lev time Ftot_0 Ftot_1 [Ftot_2] Fpres_0 Fpres_1 [Fpres_2] p_probe0(x,y) ...

* :code:`Ftot` is the momentum removed by the implicit penalty divided by :math:`\Delta t`, i.e. the pressure
  **plus** viscous force on the body (exact for a static solid).  It needs :code:`solid.implicit = 1`.
* :code:`Fpres` is :math:`\sum -p\,\mathbf{n}\,dA` over the fluid|solid faces.

The finest level must cover the whole body (use :code:`refine_box`).
Divide by :math:`q = \tfrac12\rho U^2` (times the reference length) for coefficients.
Only trust the per-step file for time behaviour; a control-volume force from a plotfile omits
:math:`d\mathbf{M}/dt` and is wrong during start-up.


.. _hydro2-moving-solid:

Moving solids
=============

Wall velocity without moving the geometry
-----------------------------------------

A body that spins or slides *along itself* does not change :math:`\phi`; only its wall velocity is
non-zero.  It enters through the solid target momentum.  For a cylinder spinning at :math:`\Omega`
(counter-clockwise positive), :math:`\mathbf{M}_s = \rho_s\,(\boldsymbol\Omega\times\mathbf{r})`:

.. code-block:: none

   solid.momentum.ic.type = expression
   solid.momentum.ic.expression.constant.Om    = 2.0      # alpha = Om R0 / U = 1
   solid.momentum.ic.expression.constant.rho_s = 100.0    # = solid.density
   solid.momentum.ic.expression.region0 = "-rho_s*Om*y"
   solid.momentum.ic.expression.region1 = "rho_s*Om*x"

The no-slip sharp wall uses :math:`\mathbf{u}_s` in its ghost state and the inert solid cells relax to it.
:code:`solid.wall_vel_surface = 1` interpolates :math:`\mathbf{u}_s` to the :math:`\phi = 0.5` point, which is
exact for rigid rotation.
(Decks: :code:`tests/FlowRotatingCylinder/UNIT_TEST_2D_Re100`, :code:`tests/FlowMovingWall/input_1D_Stokes`.)

Moving the geometry
-------------------

:code:`solid.moving = 1` moves the body itself.  The motion is **prescribed**: give
:code:`solid.phi.ic` and :code:`solid.momentum.ic` as expressions of :code:`x, y, z, t`.
At the start of every level step :code:`UpdateMovingSolid`

1. saves :math:`\phi` and re-evaluates :code:`solid.phi.ic` at the current time (analytic: the body keeps its
   shape exactly, no advection error; the expression must be periodic if the domain is);
2. re-evaluates the solid target (density, pressure, momentum -> wall velocity);
3. fills **fresh cells** (solid a step ago, fluid now) by copying every state field from the neighbour that
   was fluid before and is fluid now, the one deepest in the fluid -- one donor, all fields, so the fresh
   cell gets a thermodynamically consistent state;
4. leaves **dead cells** (fluid -> solid) alone: they become inert.

A cylinder translating at :code:`Uc` with a smooth start, re-entering through a periodic boundary:

.. code-block:: none

   solid.wall_flux = 1
   solid.moving    = 1
   solid.phi.ic.type = expression
   solid.phi.ic.expression.constant.R0  = 0.5
   solid.phi.ic.expression.constant.eps = 0.016     # ~dx/2
   solid.phi.ic.expression.constant.x0  = -3.0
   solid.phi.ic.expression.constant.Uc  = 1.0
   solid.phi.ic.expression.constant.tau = 0.5
   solid.phi.ic.expression.constant.Lx  = 12.0
   solid.phi.ic.expression.region0 = "0.5*(1.0 + tanh((sqrt((fmod(x - x0 - Uc*(t - tau*(1.0 - exp(-t/tau))) + 1000.5*Lx, Lx) - 0.5*Lx)^2 + y*y) - R0)/eps))"
   solid.momentum.ic.type = expression
   solid.momentum.ic.expression.constant.rho_s = 100.0
   solid.momentum.ic.expression.constant.Uc    = 1.0
   solid.momentum.ic.expression.constant.tau   = 0.5
   solid.momentum.ic.expression.region0 = "rho_s*Uc*(1.0 - exp(-t/tau))"
   solid.momentum.ic.expression.region1 = "0.0"

The wall must not cross more than one cell per step; if it does, a fresh cell has no fluid donor and the
solver warns *"fresh cell(s) with no fluid donor"*.

.. warning::

   The moving wall is **not conservative**: the fluid in dead cells is deleted and fresh cells are
   created (compensated on average by the wall-face flux of the moving-wall Riemann problem).
   Track the mass drift in any validation.  The body does not respond to the flow.


Validation
==========

Decks are in :code:`tests/`; each has a check script in its :code:`reference/` folder
(see :ref:`hydro2-howto-checks`).  Numbers are from :code:`bin/OPTIMIZATION_CHANGELOG.md` (Part II).

.. list-table::
   :header-rows: 1
   :widths: 26 26 48

   * - Test
     - Deck
     - Result
   * - Brinkman Couette
     - :code:`FlowCouette/input_Brinkman_Couette`
     - PASS with both walls; sharp wall within 0.014 cell, max error 0.76 %.
   * - Stokes' first problem
     - :code:`FlowMovingWall/input_1D_Stokes`
     - :code:`visc_mirror = 1`: profile error 0.40 / 0.05 / 0.01 % of U at t = 0.25 / 1.25 / 3, no-slip plane at
       0.00 dy.  :code:`visc_mirror = 0`: 9.8 / 4.4 / 2.9 %, no-slip plane at -0.50 dy.
   * - Cylinder, Re 40
     - :code:`FlowVortexShed/UNIT_TEST_2D_Re40`
     - Steady, L/D 2.31, Cd 1.607 (pressure 1.087, friction 0.520), separation 48 deg.
   * - Cylinder, Re 100
     - :code:`FlowVortexShed/UNIT_TEST_2D_Re100_kick`
     - Saturated shedding, St 0.1666, Cl amplitude 0.287, Cd 1.418.
   * - Oblique shock, 15 deg wedge
     - :code:`FlowWedge/input_Ma*`
     - Slip sharp wall with the surface normal: shock-angle error -0.09 deg at Mach 3 and 5.
   * - Viscous NACA 0012, M 0.5, Re 5000
     - generated by :code:`FlowAirfoil/reference/viscous_naca0012.py`
     - Cd 0.05484, Cl = 0 to 3e-14, separation at x/c 0.814.
   * - Rotating cylinder, Re 100
     - :code:`FlowRotatingCylinder/UNIT_TEST_2D_Re100`
     - See the table below.
   * - Galilean check, Re 40
     - :code:`FlowMixingCylinder/input_Galilean_Re40`
     - Translating body vs static: Cd +2.9 % (t = 5), +1.3 % (10), +0.4 % (15), -1.5 % (20).
   * - Cylinder through two stratified gases
     - :code:`FlowMixingCylinder/UNIT_TEST_2D`
     - Three laps, no NaN / orphan cells; gas mass drift -0.11 % (light), +0.29 % (heavy).

Rotating cylinder at Re = 100 against Stojkovic, Breuer & Durst (2002) and Kang, Choi & Lee (1999),
spin ratio :math:`\alpha = \Omega R/U`:

.. list-table::
   :header-rows: 1
   :widths: 10 30 30 30

   * - :math:`\alpha`
     - :math:`\overline{C_l}` (Numerical / reference)
     - :math:`\overline{C_d}` (Numerical / reference)
     - St (Numerical / reference)
   * - 0.5
     - -1.252 / -1.220 (+2.6 %)
     - 1.375 / 1.277 (+7.7 %)
     - 0.175 / 0.1657 (+5.6 %)
   * - 1.0
     - -2.604 / -2.504 (+4.0 %)
     - 1.240 / 1.108 (+11.9 %)
     - 0.175 / 0.1658 (+5.5 %)
   * - 2.0
     - -5.194 / -5.48 (-5.2 %)
     - 0.851 / 0.46 (+85 %)
     - 0.175 / steady


Known limitations
=================

* **Rotating-cylinder drag is high.**  Lift is within 5 % but drag is high at every spin ratio and the
  wake still sheds at :math:`\alpha = 2`, where both references find it suppressed
  (:math:`\alpha_L \approx 1.8`).  Not investigated; candidates are the wall resolution at two levels of
  refinement, the 40 D x 30 D box (the references use a 50--100 D radius), compressibility, and the
  staircase no-slip wall (:code:`solid.visc_mirror`).
* **Drag split on a staircase wall.**  On the viscous NACA 0012 the total drag is right but the
  pressure / friction split is not (0.0271 / 0.0277 against about 0.022 / 0.033): friction is
  under-resolved by the first-order staircase wall.
* **solid.visc_mirror = 2 is not adopted.**  The true-surface viscous ghost failed its gate: Re 40
  cylinder friction +9 %, away from the literature.  The plain mirror (1) acts on every step face of a
  staircase body, i.e. on up to :math:`4/\pi` of the true area (Re 40 friction 0.576 against 0.520 with 0).
  Use 1 for grid-aligned walls and isothermal walls, 0 otherwise.
* **Viscous lift deficit.**  Viscous NACA 0008 at Re 2000: lift 7--19 % below a FLUENT reference at
  AoA 2--8 deg, and it is not recovered by resolution or wall smoothness.
* **Slip-wall mass source.**  The ghost-cell slip wall has a small mass source at stagnation points
  (about 1 % of the chord mass flux at level 4, 0.3 % at level 5); the per-step force diagnostic is only
  approximate for Euler slip walls -- use a control-volume force for inviscid drag.
* **Moving solids are not conservative** and the motion is prescribed only (no fluid -> solid coupling).
  Step-to-step force noise of 0.03--0.05 in Cd comes from the fresh / covered cells.
* **solid.mu_factor > 1 is not in the viscous time-step limit.**  Fluid cells in a diffuse wall band then
  carry up to about :code:`mu_factor/2` times the fluid viscosity; the explicit viscous update can blow up.
* **Rank-count dependence.**  Sharp-wall results depend on the MPI rank count (15 deg wedge, 150 steps:
  4 vs 8 ranks differ by 2e-3 relative in density).  Pre-existing, not investigated.
* **Cold isothermal wall at a stagnation point** can blow up on an impulsive start, see
  :ref:`hydro2-ablation`.
* The embedded rigid solid is **not combined** with the deforming solid phase (:code:`elastic.on`).
