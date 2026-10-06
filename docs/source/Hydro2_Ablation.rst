.. _hydro2-ablation:

========================================================
:fas:`fire-alt;fa-fw` Hydro2: Ablating Solids
========================================================

Ablation makes a non-deforming embedded solid (:ref:`hydro2-solids`) **recede** under the heat the gas
conducts into it.  It is the standard engineering closure for a thermal-protection surface, the
heat-of-ablation (:math:`Q^*`) model, with an optional in-depth conduction model for the solid.
Default off (:code:`ablation.on = 0`); class :code:`Ablation`, :code:`src/Integrator/Ablation.H`.

.. contents::
   :local:
   :depth: 2


Model
=====

Heat-of-ablation model
----------------------

The wall is held at the ablation temperature :math:`T_w` and the heat conducted into it from the gas is
consumed by removing solid,

.. math::

   \rho_s\,Q^*\,v_\text{abl} = q_w, \qquad
   q_w = k\,\frac{\partial T}{\partial n}\bigg|_\text{gas side} \quad (q_w > 0 \text{ only}),

where :math:`Q^*` is the effective heat of ablation (sensible + latent) per unit mass of solid.
There is **no blowing**: the ablated mass is removed, not injected into the gas.

The indicator :math:`\phi` (1 fluid, 0 solid) recedes by the level-set equation

.. math::

   \frac{\partial\phi}{\partial t} = v_\text{abl}\,|\nabla\phi| ,

with first-order (Godunov) upwinding of :math:`|\nabla\phi|`.
:math:`v_\text{abl} = a\,q_w/(\rho_s Q^*)` (with :math:`a` = :code:`ablation.accel`) is known on the wall-adjacent fluid cells (from the
last RHS evaluation) and is extended off the wall by :code:`ablation.extend` averaging passes.
Cells the surface uncovers are filled exactly as for a moving solid (:ref:`hydro2-moving-solid`).

The wall heat flux that drives the recession is the flux per unit **true** surface area.  On a staircase
wall a face normal to direction :math:`d` carries :math:`q_{f,d} \approx q_n |n_d|`, so over the wall faces a
cell *has* the least-squares estimate is used,

.. math::

   q_n = \frac{\sum_d q_{f,d}\,|n_d|}{\sum_d n_d^2}, \qquad \mathbf{n} = \nabla\phi/|\nabla\phi| .

In-depth conduction
-------------------

With :code:`ablation.conduction = 1` the solid carries its own temperature :math:`T_s` (plot field
:code:`ablation_Ts`, meaningful where :math:`\phi < 0.5`),

.. math::

   \rho_s\,c_{p,s}\,\frac{\partial T_s}{\partial t} = \nabla\cdot(k_s \nabla T_s),

advanced explicitly (Euler) once per step; adiabatic at the domain boundary.
At every fluid|solid face the surface temperature follows from flux continuity between the two one-sided
gradients,

.. math::

   T_w^* = \frac{h_g T_g + h_s T_s}{h_g + h_s}, \qquad
   h = \frac{k}{\text{distance (cell centre -- surface)}},

and :code:`ablation.T_wall` becomes the **ablation temperature**:

* below it nothing is removed and the wall simply conducts, :math:`q_g = q_s`;
* once :math:`T_w^*` reaches it the surface is held at :math:`T_\text{wall}` and the excess removes solid,

.. math::

   \rho_s\,Q^*\,v_\text{abl} = q_g - q_s, \qquad
   q_g = h_g\,(T_g - T_\text{wall}), \quad q_s = h_s\,(T_\text{wall} - T_s).

:math:`Q^*` is then the heat of ablation proper; the sensible heat goes into :math:`T_s`.
The solid diffusivity :math:`\alpha_s = k_s/(\rho_s c_{p,s})` enters the viscous time-step limit
(:code:`cfl_v`).  With :code:`ablation.conduction = 0` the result is bit-identical to the model above.


Ablation inputs
===============

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Input
     - Default
     - Meaning
   * - :code:`ablation.on`
     - 0
     - 1 = surface ablation on.
   * - :code:`ablation.T_wall`
     - required
     - Wall (ablation) temperature.  Also sets :code:`solid.T_wall`.
   * - :code:`ablation.Qstar`
     - required
     - Heat of ablation per unit mass of solid.
   * - :code:`ablation.rho_s`
     - required
     - Solid density.
   * - :code:`ablation.accel`
     - 1.0
     - Recession speed multiplier (ablation is slow against the flow).
   * - :code:`ablation.extend`
     - 8
     - Number of passes extending :math:`v_\text{abl}` off the wall cells.
   * - :code:`ablation.t_start`
     - 0.0
     - No recession before this time (lets the flow establish).
   * - :code:`ablation.conduction`
     - 0
     - 1 = in-depth conduction in the solid; :code:`T_wall` is then the ablation temperature.
   * - :code:`ablation.k_s`
     - required if conduction
     - Solid thermal conductivity.
   * - :code:`ablation.cp_s`
     - required if conduction
     - Solid specific heat.
   * - :code:`ablation.T_solid0`
     - :code:`ablation.T_wall`
     - Initial solid temperature.

Ablation needs three other things switched on:

.. code-block:: none

   apply_embedded_solid = 1
   solid.wall_flux      = 1        # the sharp wall
   solid.visc_mirror    = 1        # no-slip plane ON the wall face (isothermal wall in a stream)
   thermal.conduction   = 1        # heat conduction in the gas
   thermal.Pr           = 0.72

   ablation.on      = 1
   ablation.T_wall  = 2.4876e-3    # cold wall at the free-stream temperature
   ablation.rho_s   = 1.0
   ablation.Qstar   = 80.0
   ablation.t_start = 1.0          # let the flow establish

and, for a conducting solid (:code:`tests/FlowAblation/input_1D_Stefan2`),

.. code-block:: none

   ablation.T_wall     = 0.99         # ablation temperature
   ablation.Qstar      = 0.08
   ablation.rho_s      = 1.0
   ablation.conduction = 1            # in-depth conduction in the solid
   ablation.k_s        = 4.8611111e-3
   ablation.cp_s       = 3.5
   ablation.T_solid0   = 0.985

.. tip::

   To test the wall **heat flux** alone, make :code:`ablation.Qstar` huge: the surface does not move
   (:code:`input_2D_BluntBody`).  To test the fluid|solid thermal **contact** alone, put
   :code:`ablation.T_wall` out of reach (:code:`input_1D_Contact`).

If the surface moves more than half a cell per step the solver warns
*"ablation: surface moves ... cells per step (reduce ablation.accel)"*.

Plot fields: :code:`ablation_q` (heat flux into the wall per unit true surface area, on wall-adjacent fluid
cells), :code:`ablation_v` (extended recession speed), :code:`ablation_Ts` (solid temperature).


Verification
============

Decks in :code:`tests/FlowAblation`, check scripts in :code:`tests/FlowAblation/reference`.

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Test
     - Deck
     - Result
   * - One-phase Stefan problem
     - :code:`input_1D_Stefan`
     - :math:`s = 2\lambda\sqrt{\alpha t}`.  At :math:`\Delta T/T` = 1 %: recession error at t = 7.2 of
       -2.0 / -0.7 / -0.2 % on 200 / 400 / 800 cells (order about 1.6), wall heat flux +0.2 %.
   * - Two half-spaces in contact
     - :code:`input_1D_Contact`
     - Surface temperature :math:`T_i = (e_g T_g + e_s T_s)/(e_g + e_s)`, :math:`e = \sqrt{k\rho c}`: +0.02 % of
       :math:`\Delta T`; profiles 0.17 % (gas), 0.02 % (solid).
   * - Two-phase Stefan problem
     - :code:`input_1D_Stefan2`
     - Recession -0.1 ... +0.4 %, net flux within 0.5 %, profiles 0.13 % (gas), 0.01 % (solid).
   * - Gap conduction
     - :code:`input_1D_GapConduction`
     - Wall flux equals :math:`k\,dT/dx` of the gas next to the wall to 0.35 %.
   * - Inclined wall (26.6 deg)
     - :code:`input_2D_InclinedStefan` vs :code:`input_1D_ClosedGap`
     - +0.4 ... +1.2 % against the aligned run; face flat to 0.01 cell.
   * - Receding cylinder
     - :code:`input_2D_Cylinder`
     - +1.8 / 0.0 / -0.8 % at t = 1.5 / 3 / 4; out-of-round 0.03 cells rms.
   * - Mach 2 flat plate, isothermal
     - :code:`input_2D_FlatPlate`
     - With :code:`visc_mirror = 1`: :math:`q_w` +2.1 / +1.1 / +0.8 % at x = 0.1 / 0.4 / 0.8 (Eckert reference
       temperature), :math:`c_f` +6.0 / +2.8 / +2.6 %, :math:`q_w \sim x^{-0.504}`.
   * - Mach 3 blunt body, cold wall
     - :code:`input_2D_BluntBody`
     - Stagnation pressure -0.36 % (Rayleigh pitot); stagnation heat flux +9.8 % (Fay--Riddell).
   * - Ablating Mach 3 wedge
     - :code:`UNIT_TEST_2D_Wedge`
     - Tip blunts: nose radius 0.0052 -> 0.0133, peak heat flux 5.14 -> 2.64; no closed form.

.. note::

   The one-phase Stefan solution assumes constant density.  With the deck's 10 % colder wall the gas at
   the wall is 11 % denser and the wall heat flux reads +2.3 % against the reference *at every
   resolution*; that bias is the reference, not the solver (it drops to +0.2 % at
   :math:`\Delta T/T` = 1 %: :code:`ablation.T_wall=0.99 ablation.Qstar=0.1757`).


Known limitations
=================

* **Cold-wall blow-up on an impulsive start at high conductivity (open bug).**
  Mach 3 blunt body with :math:`\mu = 10^{-2}` (7 cells per boundary-layer scale): the run blows up at
  t = 0.0078 -- the temperature of the second fluid cell on the stagnation line grows exponentially until
  the HLLC solver fails.  It happens *during the impulsive start*, before a bow shock exists.  Same instant
  for :code:`cfl_v` 0.3 to 0.15 (not a time-step limit), with :code:`solid.visc_mirror` 0 or 1, and with a
  sharp 0/1 :math:`\phi`.  It is **stable** with an adiabatic wall, with a hot wall (:math:`T_w = T_0`), with
  :code:`thermal.Pr = 7.2` (10x less conduction), with :math:`\mu = 2\times10^{-3}`, and with a soft start
  (gas initially at rest, inflow-driven).  So: strong conduction to a cold isothermal wall at a stagnation
  point.  Not diagnosed.  The :math:`\mu = 2\times10^{-3}` run stays the reference; a soft start gets past
  this instant (those runs later died where the bow shock leaves the box, a separate problem).
* **Single level only.**  Ablation runs with :code:`amr.max_level = 0`.
* **No blowing.**  The ablated mass is removed, not injected into the gas.
* **Under-resolved tips.**  On the Mach 3 wedge the nose boundary layer is below one cell and the tip
  heating is under-resolved (about -30 % against Fay--Riddell for the measured nose radius).
  Wall-cell heat flux scatters +-8 % cell to cell around a curved body (staircase).
* **Start-up lag.**  The recession lags by about 0.07 cell at start-up.
* The recession is not conservative (cells are uncovered and filled as for :code:`solid.moving`).
