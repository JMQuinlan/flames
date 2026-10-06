.. _hydro2-elastic:

=======================================================
:fas:`cubes;fa-fw` Hydro2: Deforming (Elastic) Solids
=======================================================

A solid that deforms is not an embedded boundary -- it is one of the two *phases* of the 6-equation
model.  With :code:`elastic.on = 1` a phase that is given a shear modulus becomes a hyperelastic solid
following Favrie, Gavrilyuk & Saurel, *Solid--fluid diffuse interface model in cases of extreme
deformations*, JCP 228 (2009) 6037, the solid extension of the Saurel--Petitpas--Berry model this
solver already uses.  One phase can be a solid (metal|gas, soft solid in a gas) or both can be
(an Al|Cu contact), sharing one deformation field.

The implementation lives in :code:`src/Integrator/Hydro2_Solid.H` (class :code:`ElasticSolid`;
:code:`ElasticSolid.H` only forwards to it) and
:code:`src/Solver/Local/FluidRiemann/HLLC_Elastic.H`.
Nothing here runs unless :code:`elastic.on = 1`, and with it off the solver is bit-identical.

.. contents::
   :local:
   :depth: 2


Model
=====

The **hydrodynamic part** of the solid is the phase's own stiffened-gas EOS, unchanged.
The **elastic part** is carried by the averaged cobasis :math:`\mathbf{E}^b = \alpha_s^{1/3}\nabla X_b`
(the gradients of the Lagrangian coordinates), from which the averaged Finger tensor is built,

.. math::

   \mathbf{G} = \sum_b \mathbf{E}^b \otimes \mathbf{E}^b, \qquad
   \tilde{\mathbf{G}} = \mathbf{G}/|\mathbf{G}|^{1/3} .

The cobasis is transported with the mixture velocity,

.. math::

   \frac{\partial E^b_i}{\partial t} + v_k \frac{\partial E^b_i}{\partial x_k}
   + E^b_k \frac{\partial v_k}{\partial x_i} = 0 ,

and gives the mixture stress :math:`\boldsymbol\sigma = -p\,\mathbf{I} + \mathbf{S}` with

.. math::

   \mathbf{S} = -\Big(\sum_k \mu_k \frac{\alpha_k\rho_k}{\rho_{k0}}\Big)\,
   \mathrm{dev}\big(\tilde{\mathbf{G}}^2 - \tilde{\mathbf{G}}\big),
   \qquad
   W = \Big(\sum_k \mu_k \frac{\alpha_k\rho_k}{\rho_{k0}}\Big)\,
   \tfrac14\,\mathrm{tr}\big((\tilde{\mathbf{G}} - \mathbf{I})^2\big) .

:math:`\mu_k` is the shear modulus and :math:`\rho_{k0}` the unstrained density of phase :math:`k`.
The transported phase mass is used in the prefactor, so a trace of solid in the fluid carries no stress.
The elastic energy per volume :math:`W` is part of the conserved total energy: the hydrodynamic internal
energy is :math:`\rho E - \tfrac12\rho|\mathbf{u}|^2 - W`.
In the small-strain limit :math:`\mathbf{S} \to 2\mu\,\mathrm{dev}(\boldsymbol\varepsilon)` (Hooke).

The Riemann solver is the HLLC solver of Favrie Sec. 4.2 (:code:`hllc_elastic`): the normal stress
:math:`\sigma_{11} = -p + S_{11}` replaces the pressure in the contact speed and the star states, and the
longitudinal speed in the wave-speed estimate and the CFL condition is

.. math::

   c_l^2 = c^2 + \tfrac43\,\frac{\mu_0\alpha_0 + \mu_1\alpha_1}{\rho} .

**Shear is transmitted only across solid|solid faces** (both sides with a solid volume fraction above
:code:`elastic.solid_cut`).  Otherwise the star shear stress is zero: a fluid slides on a solid, and a
solid surface next to a fluid is traction-free in shear.

In 2D (plane strain) the field :code:`ebasis` has five components,
:math:`E^1 = (c_0, c_1)`, :math:`E^2 = (c_2, c_3)`, :math:`E^3 = (0, 0, c_4)`;
:math:`E^3_z = \alpha_s^{1/3}` is carried because it is not 1 across a diffuse interface.
In 3D it has all nine.


Elastic-solid inputs
====================

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Input
     - Default
     - Meaning
   * - :code:`elastic.on`
     - 0
     - 1 = hyperelastic solid phase(s) on.  Adds the cobasis field and switches the Riemann solver to
       :code:`hllc_elastic`.
   * - :code:`elastic.mu0`
     - 0
     - Shear modulus of phase 0 (:math:`\eta = 1`).  0 = that phase is a fluid.
   * - :code:`elastic.mu1`
     - 0
     - Shear modulus of phase 1 (:math:`\eta = 0`).  0 = that phase is a fluid.
   * - :code:`elastic.rhoref0`
     - 1
     - Unstrained (reference) density of phase 0, used if :code:`mu0 > 0`.
   * - :code:`elastic.rhoref1`
     - 1
     - Unstrained (reference) density of phase 1, used if :code:`mu1 > 0`.
   * - :code:`elastic.cobasis.ic.type`
     - absent
     - Optional pre-strained initial state: :code:`constant` or :code:`expression` with
       :code:`region0..` = :math:`e^1_x, e^1_y, e^2_x, e^2_y, e^3_z` (2D) or the nine :math:`e^b_i` (3D).
       :math:`\det(e)` should equal :math:`\rho_s/\rho_{s0}`.  Its elastic energy is added to the initial :math:`\rho E`.
   * - :code:`elastic.fluid_cut`
     - 0.1
     - Cells with a solid volume fraction below this are fluid: their cobasis is reset to the isotropic
       unstrained value every ghost fill.  <= 0 disables.
   * - :code:`elastic.solid_cut`
     - 0.5
     - A face transmits shear only if the solid volume fraction exceeds this on both sides.
   * - :code:`elastic.reconstruct`
     - 1
     - 1 = second order: MC-limited linear reconstruction of the cobasis at faces where the cell and both
       neighbours are solid; 0 = zero slope everywhere (the paper's first-order scheme, App. B).
   * - :code:`elastic.reconstruct_cut`
     - 0.99
     - The reconstruction is only applied where the cell and both neighbours have a solid volume fraction
       above this (per-component limiting is not rotation invariant across the smeared surface).
   * - :code:`elastic.objective_shear`
     - 1
     - 1 = the upwind shear dissipation at solid|solid faces acts on the symmetric velocity gradient, so it
       does not brake a rigid rotation; 0 = the paper's 1D form.

The boundary condition of the cobasis is not an input: it is zero-gradient on physical boundaries and
periodic where :code:`geometry.is_periodic` is set.

Why the fluid cells are reset
-----------------------------

The cobasis is transported everywhere, and in the fluid it is stretched and rotated without bound by
shear layers and vortices.  Upwinding then feeds that into the solid's surface cells, the stress and the
density blow up there, and the time step collapses.  Fluid cells have no deformation memory, so wherever
the solid volume fraction is below :code:`elastic.fluid_cut` the cobasis is reset to the unstrained
isotropic value -- an instantaneous shear relaxation in the fluid.  With :code:`0.01` the shock / cylinder
run completes but leaves hot gas pockets in the wake; with the default :code:`0.1` there are none.


Setting up a case
=================

Soft solid in a gas (the :code:`UNIT_TEST_2D_ShockCylinder` settings):

.. code-block:: none

   Riemann_Solver.type = hllc_elastic
   Limiter.type        = minmod
   nghost              = 4
   integration.type    = RungeKutta
   integration.rk.type = 3

   elastic.on      = 1
   elastic.mu0     = 50.0     # phase 0 (eta = 1): the solid;  phase 1 is the gas (mu1 = 0)
   elastic.rhoref0 = 10.0

   eos0.gamma = 4.4           # hydrodynamic part of the solid: stiffened gas
   eos0.p0    = 100.0
   eos1.gamma = 1.4
   eos1.p0    = 0.0

The solid body is drawn with :code:`eta.ic` exactly like a liquid would be; leave a small trace of the other
phase on both sides as the test decks do (:code:`1.0e-6 + (1.0 - 2.0e-6)*(...)`; the copper deck uses
:code:`1.0e-4` as in the paper).
Set :code:`density0.ic` to :code:`elastic.rhoref0` for a stress-free start.
A phase with :code:`elastic.mu<k> = 0` behaves as the plain fluid, which is how each test is run twice
(elastic and liquid) for comparison.

.. note::

   If :code:`elastic.on = 1` and another Riemann solver is requested, it is replaced by
   :code:`hllc_elastic` with a console message: the solid stress only exists in that flux.

Metal|gas interfaces
--------------------

Copper in air is a different regime: :math:`\pi/p \approx 3.4\times10^5` and a density ratio of 8900.
See :ref:`hydro2-elastic-limits` before choosing a limiter; the copper plate impact deck
(:code:`input_2D_PlateImpact_Copper`) ships with :code:`Limiter.type = godunov`.


Verification
============

Decks :code:`tests/FlowElasticSolid/input_*` and :code:`UNIT_TEST_2D_*`; checks in
:code:`tests/FlowElasticSolid/reference/*_check.py`.

.. list-table::
   :header-rows: 1
   :widths: 24 26 50

   * - Test
     - Deck
     - Result
   * - Elastic waves
     - :code:`input_1D_ElasticWaves`
     - :math:`c_l` +0.21 %, :math:`c_t` +0.39 %; plateaux within 0.06 % of linear theory.
   * - Al|Cu shock tube
     - :code:`input_1D_CuAlShock`
     - Within 0.03 %; :math:`\sigma_{11}` continuous to 1e-5 across the contact.
   * - Gas shock on a soft solid
     - :code:`input_1D_ShockImpact`
     - Stresses within 0.5 % of impedance matching.
   * - Finite-strain piston
     - :code:`input_1D_Piston`
     - 7--26 % compression: all quantities within 0.09 % of the exact Rankine--Hugoniot state.
   * - Homogeneous finite deformation
     - :code:`input_Homogeneous`
     - Every stress component and :math:`W` equal the closed form to 1e-6.
   * - Five-wave Riemann problem (copper)
     - :code:`input_1D_FiveWave`
     - Within 0.4 %.
   * - Rigid rotation of a stress-free disc
     - :code:`input_2D_RotatingDisc`
     - Angular momentum -0.41 % after 90 deg (-3.0 % with the first-order scheme).
   * - Oblique plane waves
     - :code:`input_2D_ObliqueWave`
     - Loss per period 0.88 / 0.71 % (longitudinal / shear).
   * - Zoeppritz coefficients, Al|Cu
     - :code:`input_2D_ObliqueInterface`
     - Reflected P +0.1 %, reflected SV +1.8 %, transmitted P +0.8 %, transmitted SV +14.8 % (coefficient 0.025).
   * - Line source
     - :code:`input_2D_LineSource`
     - Front position exact to a cell; peak -7.9 % at 400 x 400.
   * - Mach 2 shock on a cylinder
     - :code:`UNIT_TEST_2D_ShockCylinder`
     - Elastic body keeps its shape (0.396 x 0.396 -> 0.392 x 0.404), mass drift 0.0000 %.
   * - Projectile on plate
     - :code:`UNIT_TEST_2D_PlateImpact`
     - Projectile and plate stay bonded; mass drift -0.0013 %.
   * - Copper plate impact
     - :code:`input_2D_PlateImpact_Copper`
     - Plate bends and springs back as in Favrie et al. (2009) Fig. 16 (first order, 400 x 400).


.. _hydro2-elastic-limits:

Known limitations
=================

* **Vibrating slab test fails.**  :code:`input_1D_SlabVibration` blows up at t = 1.14 of 1.2 with the
  second-order solid.  It is a base-solver interface instability next to a light gas (it also occurs with
  :code:`elastic.on = 0` and :code:`hllc`): a stiff slab oscillating normally next to a gas at a density
  ratio of 100 or 1000 grows a transverse instability in the interface cells of a 1D problem.  Density
  ratio 10 is clean.  With :code:`elastic.fluid_cut = 0` the trace solid's rigidity suppresses it, but the
  2D cases need :code:`fluid_cut > 0`, so the two requirements conflict for light gases.
* **Higher-order limiters at metal|gas interfaces.**  Solid|gas interface advection is exact to 1e-9
  with :code:`Limiter.type = godunov`.  With a soft solid the higher-order limiters are stable but leave
  1--2 % velocity noise and a 3--7 % residual stress at the interface (:code:`thinc` worse).  At a
  copper|air interface every higher-order limiter was unstable with the default settings, and equally with
  :code:`elastic.on = 0` -- a base-solver limit, not the elastic terms.  In particular the WENO3
  reconstruction is not equilibrium-preserving there.  The mismatch is between the first-order
  **donor-cell** :math:`\alpha` row and the **limiter-reconstructed** phasic energies; setting
  :code:`eta_consistent_advect = 1`, which takes the :math:`\alpha` face value from the same reconstruction,
  removes it.  See :ref:`hydro2-howto-limiter`, and note the caveat there:
  :code:`eta_consistent_advect = 1` is **not** the validated setting for bubble collapse.
* **Single level only.**  Average-down / :code:`PostAverageDown` do not debit :math:`W`; run with
  :code:`amr.max_level = 0`.
* **Sliding at solid|fluid interfaces.**  The paper's ghost-fluid correction (Sec. 6) is not implemented;
  thin filaments leave the top / bottom of the body in the shock / cylinder test (the paper reports the
  same without the correction).
* **Rotation.**  The disc's spin still falls 0.50 -> 0.463 over 90 deg: the smeared rim gains inertia at
  conserved angular momentum.
* No plasticity; no reflecting BC for the cobasis; not combined with the embedded rigid solid
  (:code:`solid.*`); 3D is compile- and wave-tested only (1D waves in a 400 x 4 x 4 box within 1--2 %).
* The copper plate impact is first order and 2.5x coarser than the paper's 1000 x 1000 second-order run, so
  the interfaces are diffuse and the internal stress-wave detail is missing.
