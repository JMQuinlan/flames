.. _hydro2-shell:

==========================================================
:fas:`circle-notch;fa-fw` Hydro2: Polymer / Lipid Shells
==========================================================

A coated interface -- the polymer or lipid shell of a contrast microbubble -- is modelled as a
**visco-elastic surfactant carried by the diffuse interface**.
Two pieces make it up, and both ride on the existing capillary (CSF) source:

* an **elastic** part: the piecewise effective surface tension of Marmottant et al.
  (JASA 118, 2005, Eq. 4) with buckling and rupture, switched on by :code:`marmottant = 1`;
* a **viscous** part: the Boussinesq--Scriven interfacial viscosity, :code:`shell.kappa_s`
  (dilatational) and :code:`shell.mu_s` (shear).

With :code:`marmottant = 0` and :code:`shell.kappa_s = shell.mu_s = 0` (the defaults) the interface is a
bare one with constant :code:`sigma`.

.. note::

   :code:`tests/FlowMarmottant/MARMOTTANT_MODEL.md` describes the *first* implementation, in which the
   bubble radius was recovered from the interface-averaged curvature, :math:`\sigma(R)`.
   That path is gone.  The solver now uses the **advected-shell** form described here; there is no
   curvature anywhere in the tension.

.. contents::
   :local:
   :depth: 2


Model
=====

Advected shell
--------------

The shell tension is a function of the monolayer's **areal strain** -- a material property carried *by*
the interface -- not of the local curvature.  A conserved shell density (shell material per unit volume)
is therefore transported with the flow,

.. math::

   c = \Gamma\,|\nabla\eta|, \qquad
   \frac{\partial c}{\partial t} + \nabla\cdot(c\,\mathbf{u}) = 0, \qquad
   \Gamma = \frac{c}{|\nabla\eta|},

where :math:`\Gamma` is the shell material per unit *area* (plot fields :code:`shell` = :math:`c` and
:code:`Gamma`).  The shell is initialised to :math:`c = |\nabla\eta|`, i.e. :math:`\Gamma_0 = 1`:
**the shell starts unstrained at** :code:`marmottant.R0`.

The effective tension is

.. math::

   \sigma(\Gamma) =
   \begin{cases}
   0 & \Gamma \ge \Gamma_\text{buck} \quad\text{(buckled)} \\[2pt]
   \chi\left(\dfrac{\Gamma_\text{buck}}{\Gamma} - 1\right) & \Gamma < \Gamma_\text{buck} \quad\text{(elastic)} \\[8pt]
   \sigma_\text{water} & \chi\,(\Gamma_\text{buck}/\Gamma - 1) \ge \sigma_\text{break} \quad\text{(ruptured)}
   \end{cases}
   \qquad
   \Gamma_\text{buck} = \left(\frac{R_0}{R_\text{buckling}}\right)^2 .

:math:`\Gamma_\text{buck}/\Gamma` is exactly Marmottant's area ratio :math:`A/A_\text{buck} = (R/R_\text{buckling})^2`,
so this is the same :math:`\sigma(A)` law -- but with the area carried by the shell instead of re-derived
from curvature every step.  The input :code:`sigma` is reused as :math:`\sigma_\text{water}`, the bare
(post-rupture) tension.  At start-up the solver prints the derived :math:`\Gamma_\text{buck}` and

.. math::

   \sigma(t = 0) = \chi\,(\Gamma_\text{buck} - 1) .

What this buys over :math:`\sigma(R)`:

* no curvature in the tension, so nothing to amplify or filter;
* shell material is conserved, so stretching genuinely dilutes :math:`\Gamma` and the elastic / rupture
  branches are reached for the right reason;
* :math:`\Gamma` is uniform *across* the band by construction (:math:`c` and :math:`|\nabla\eta|` share one
  profile), so the band cannot split into "buckled" inner cells and "elastic" outer cells;
* non-spherical shells and several bubbles need no labelling.

Shell viscosity
---------------

The full surface stress of a Newtonian interface (Boussinesq--Scriven) is

.. math::

   \mathbf{T}_s = \sigma(\Gamma)\,\mathbf{P} + (\kappa_s - \mu_s)\,(\nabla_s\!\cdot\mathbf{u})\,\mathbf{P}
   + 2\mu_s\,\mathbf{D}_s, \qquad
   \mathbf{P} = \mathbf{I} - \mathbf{n}\otimes\mathbf{n}, \quad
   \nabla_s\!\cdot\mathbf{u} = \mathrm{tr}(\mathbf{P}\,\nabla\mathbf{u}),

and the interfacial force is its divergence with the same :math:`|\nabla\eta|` normalisation the capillary
tensor already uses, so the two add.  The dilatational part collapses into the scalar the capillary
tensor already multiplies,

.. math::

   \sigma_\text{tot} = \sigma(\Gamma) + (\kappa_s - \mu_s)\,\nabla_s\!\cdot\mathbf{u} .

For a sphere, :math:`\nabla_s\!\cdot\mathbf{u} = 2\dot R/R` and :math:`\mathbf{D}_s = (\dot R/R)\,\mathbf{P}`, the shear
viscosity cancels identically, and the normal balance gives

.. math::

   \Delta p = \frac{2\sigma}{R} + \frac{4\kappa_s \dot R}{R^2},

the Marmottant shell-damping term.  :math:`\kappa_s` **is therefore the whole of the shell viscosity for
spherical dynamics**; :math:`\mu_s` only acts on shape change at constant area.

Time-step limit
---------------

The dilatational term is a *surface* momentum diffusion with kinematic diffusivity

.. math::

   \nu_s = \frac{\kappa_s\,|\nabla\eta|}{\rho} \sim \frac{\kappa_s}{\rho\,\epsilon} .

Because the interface width is held at about a cell, :math:`\nu_s \sim 1/\Delta x` and the limit scales as
:math:`\Delta t \sim \Delta x^3`, while the acoustic limit only scales as :math:`\Delta x`.
**Refinement makes the shell viscosity the binding constraint** even though it is invisible on a coarse
grid.  It is included in the viscous limit (:code:`cfl_v`) automatically.


Shell inputs
============

Elastic shell
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Input
     - Default
     - Meaning
   * - :code:`apply_surface_tension`
     - false
     - Must be on: the shell acts through the capillary term.
   * - :code:`sigma`
     - 0
     - Surface tension; with :code:`marmottant = 1` the bare (post-rupture) tension :math:`\sigma_\text{water}` [N/m].
   * - :code:`marmottant`
     - 0
     - 0 = constant :code:`sigma`; 1 = Marmottant :math:`\sigma(\Gamma)`.
   * - :code:`marmottant.R0`
     - 0 (required > 0)
     - Reference radius: the shell is unstrained (:math:`\Gamma = \Gamma_0 = 1`) here [m].
   * - :code:`marmottant.R_buckling`
     - 0 (required > 0)
     - Buckling radius [m].
   * - :code:`marmottant.chi`
     - 0
     - Shell elastic modulus :math:`\chi` [N/m].
   * - :code:`marmottant.sigma_break`
     - 1e30
     - Rupture tension [N/m].  The default (huge) means the shell never ruptures.

Shell viscosity
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Input
     - Default
     - Meaning
   * - :code:`shell.kappa_s`
     - 0
     - Surface dilatational viscosity :math:`\kappa_s` [kg/s].
   * - :code:`shell.mu_s`
     - 0
     - Surface shear viscosity :math:`\mu_s` [kg/s].  Cancels for spherical motion.
   * - :code:`shell.kappa_s_subcycle`
     - 0
     - 1 = advance the :math:`\kappa_s` stress in its own sub-cycle on the interface band instead of letting
       its :math:`\Delta x^3` limit throttle the global time step.  See the limitations.
   * - :code:`shell.kappa_s_subcycle_max`
     - 512
     - Hard cap on the number of sub-steps (runaway guard).
   * - :code:`shell.kappa_s_verbose`
     - 0
     - Print the sub-step count and :math:`\nu_s` each sub-cycle.
   * - :code:`shell.nus_rho_floor`
     - 1.0
     - Density floor for the :math:`\nu_s` stability estimate (not for the physics).
   * - :code:`shell.divs_kinematic`
     - 0
     - 1 = take the normal part of the surface dilatation from the interface's own kinematics.  More
       accurate on the linear damping test but **unstable** at three levels of refinement; leave at 0.

Numerics
--------

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Input
     - Default
     - Meaning
   * - :code:`shell_gate_free`
     - 1
     - 1 = no grid-scaled freeze of :math:`\Gamma`; the consistent projector kills the bulk source.
   * - :code:`shell_bulk_extend`
     - 1
     - 1 = extend :math:`\Gamma` from the band into the adjacent bulk (stops a moving interface eating
       unstrained shell).
   * - :code:`capillary_use_eta`
     - 1
     - 1 = capillary tensor built on :math:`\eta`; 0 = on the advected colour function (legacy, slated for removal).
   * - :code:`omega_sym_mirror`, :code:`sym_face_central`
     - 1
     - Symmetry-plane treatments of the capillary / shell stress, see :ref:`hydro2-bc`.

.. note::

   The shell keys are a mix of dotted (:code:`shell.kappa_s`, :code:`marmottant.chi`) and underscored
   (:code:`shell_gate_free`, :code:`shell_bulk_extend`) names.  They are spelled here exactly as the
   parser reads them.

Example
-------

A 2 :math:`\mu`\ m coated microbubble (:code:`tests/FlowMarmottant/input_Sch20-Oscillating_Marmottant`):

.. code-block:: none

   apply_surface_tension  = 1
   sigma                  = 0.0728    # post-rupture / bare interface tension [N/m]

   marmottant             = 1
   marmottant.R0          = 2.0e-6    # reference radius (shell strain measured from here)
   marmottant.R_buckling  = 1.96e-6   # 0.98 R0 -> Gamma_buck = 1.0412, sigma(R0) = 0.0412 N/m
   marmottant.chi         = 1.0       # shell elastic modulus [N/m]
   marmottant.sigma_break = 0.073     # rupture tension [N/m]

   shell.kappa_s = 7.2e-9             # surface dilatational viscosity [kg/s]
   shell.mu_s    = 0.0

The bubble starts mid-elastic and buckles on the inward stroke.  The elastic window of a real shell is
intrinsically narrow, :math:`R_\text{rupture}/R_\text{buckling} = \sqrt{1 + \sigma_\text{break}/\chi} = 1.0359`,
so a large-amplitude free collapse spends most of its time buckled (:math:`\sigma = 0`).  That is the
physical "compression-only" behaviour, not a defect of the setup.

.. tip::

   **Size the bubble for the shell.**  The shell enters the normal stress balance as :math:`2\sigma/R` and
   :math:`4\kappa_s\dot R/R^2`.  At a 20 mm radius a physical shell is dynamically invisible
   (:math:`2\sigma/R_0 = 7.3` Pa); at 2 :math:`\mu`\ m it is 0.73 of the ambient pressure.

Bubble time series
------------------

Sampling a fast bubble oscillation from 3D plotfiles is not affordable.  Setting :code:`amr.thermo.plot_int`
or :code:`amr.thermo.plot_dt` writes :code:`gas_volume`, :code:`gas_pressure_int`, :code:`kinetic_energy` and
:code:`interface_area` to :code:`thermo.dat` at the cost of one masked pass over the grid per sample.
:code:`plot_minimal = 1` keeps the plotfiles small.


Verification
============

Decks in :code:`tests/FlowMarmottant`; analysis scripts in :code:`tests/FlowMarmottant/reference`.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Deck
     - What it tests
   * - :code:`input_Marmottant_Laplace_Static`
     - Static coated bubble in Laplace balance with :math:`\sigma(\Gamma)`.
   * - :code:`input_Linear_ShellDamping_UNIT`
     - Isolates :math:`\kappa_s` with an analytic target: the amplitude of a small free oscillation decays as
       :math:`e^{-\beta t}`, :math:`\beta = 2\mu/(\rho R_0^2) + 2\kappa_s/(\rho R_0^3)`.  Run with bulk viscosity
       and with :math:`\kappa_s` at the same analytic :math:`\beta`; geometry cancels in the ratio.
   * - :code:`input_Sch20-Oscillating_Marmottant`
     - Coated twin of the Schmidmayer et al. (2020) low-pressure-ratio case.  The minimum radius
       discriminates the shell terms: reference ODE :math:`R_\text{min}/R_0` = 0.1908 (uncoated), 0.3156
       (elastic only), 0.3906 (viscous only), 0.5172 (full shell).
   * - :code:`input_Sch20-Collapsing_Marmottant`
     - Coated twin of the high-pressure-ratio (collapsing) case.
   * - :code:`input_TwoBubble_Marmottant_2D`
     - Two coated bubbles of different radii, each in its own Laplace balance (local :math:`\sigma`
       discriminator; no labelling needed with the advected shell).
   * - :code:`input_non_spherical_decay/input_l2_mus*`
     - Decay of an :math:`l = 2` shape mode at :math:`\mu_s` = 0, 1e-6, 5e-6: the one test in the suite that
       can see the surface **shear** viscosity at all (every spherical case is blind to it).

Measured on the linear shell-damping test, :math:`\kappa_s` damping (measured / analytic) with the default
band-velocity dilatation: 0.52 at :math:`R_0/\Delta x = 8` and about 0.68 at 16.


Known limitations
=================

* **Sub-cycling fails at five levels of refinement.**  :code:`shell.kappa_s_subcycle = 1` is validated at
  three levels (trajectory within 0.05 % of the time-step-limited path, 10.7x faster) but at five levels
  the required sub-step count climbs past the cap because :math:`\nu_s` grows during the run, and once
  clamped the sub-step is no longer stability-bounded.  The time-step-limited path (0) is the supported one.
* **Kinematic dilatation is unstable.**  :code:`shell.divs_kinematic = 1` is closer to the analytic damping
  (0.95--1.07 at :math:`R_0/\Delta x = 16`) but divides by :math:`|\nabla\eta|` twice and goes unstable at
  band-edge cells on a three-level run.
* **The default dilatation under-predicts the damping** on a coarse interface (see the numbers above).
* **Rupture is stateless.**  :math:`\sigma` jumps to :math:`\sigma_\text{water}` once the elastic value would
  exceed :code:`sigma_break`; there is no hysteresis (a ruptured shell heals if it is compressed again).
* **Shear viscosity.**  :code:`shell.mu_s` is applied in the stress build (:math:`2\mu_s\mathbf{D}_s`), although
  some comments in the source still describe it as not implemented.  Only the :math:`l = 2` shape-mode decks
  exercise it; no pass / fail number for them is recorded in the change log.
* :code:`tests/FlowMarmottant/input_Laplace_Marmottant_2D` aborts at :code:`Parse`: the deck predates
  :code:`marmottant.R0`.
* The colour function :code:`cfun` (:code:`capillary_use_eta = 0`) is dead code awaiting removal.
