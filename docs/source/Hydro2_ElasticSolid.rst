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


Elastic solid model
===================

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

A second solver, :code:`Riemann_Solver.type = hllc_solid`, is the reduced three-wave HLLC of Favrie & Schmidmayer
(2026), Sec. 4.1.1.  It is selected by name only (the default stays :code:`hllc_elastic`) and differs in three
places: the shear stress and the tangential velocity are continuous across the contact at *every* face (their
eqs. 82 and 84; :code:`hllc_elastic` does this only between two solid cells and lets a fluid slide on a solid), the
star phase pressures follow the isentrope (eq. 93), and there is no objective-shear correction.  The parameter
:code:`Riemann_Solver.zeta` (default 1) is the paper's safety factor on the elastic part of the sound speed.  On the
copper--aluminium impact of that paper both solvers give the same result (within 0.15 % of the published plateaux).
Because the shear treatment also applies where a side is fluid, fluid shear layers are diffused by this solver;
do not use it for fluid-only problems.

Several solids in the solid phase.  With species (:code:`species.eos = mixture`, :code:`species.phaseK.pinf`,
:code:`species.phaseK.rho_ref`) and :code:`elastic.species.mu`, :code:`elastic.species.sigma_y`,
:code:`elastic.species.Ky` (one value per species) the solid phase can hold several metals, each with its own
equation of state, shear modulus, yield stress and plastic viscosity, and the other phase is free for a gas.  The
titanium spallation of Favrie & Schmidmayer (2026), case 1 (:code:`input_1D_TiSpallation`), runs this way: the air
trace inside the titanium opens the spall.  The deck needs :code:`cutoff = 1e-8` so that the :math:`10^{-4}` air
trace is relaxed.  The deck uses a titanium yield stress of 1030 MPa (Ndanou et al. 2015, Table 1; the 530 MPa
printed in Table 3 of the 2026 paper gives half the published elastic precursor).  With it the wave fronts agree
with the published figure to 0.002 cm at 0.8 and 1.2 microseconds; the plastic-wave pressure is 7 % low and the
plate breaks earlier and into one more fragment than in the figure.  Pressure relaxation is instantaneous by default (the paper's "first approximation" and the model of Ndanou et al.
2015).  :code:`relax.omega` (default negative = instantaneous) switches on the finite-rate law
:math:`\dot\alpha_1 = \omega\,\alpha_1\alpha_2\,(p_1 - p_2)` of the paper.  The volume fraction then covers only
part of the way to equilibrium in a step, the phases keep their own pressures, the volume change does work at the
interface pressure :math:`p_I = (Z_2 p_1 + Z_1 p_2)/(Z_1+Z_2)` and the remaining defect of the total energy is shared
in proportion to the volume fractions (the paper's eqs. 113--115).  With :code:`relax.energy_share = mass` the defect is shared by mass
fraction instead (an alternative the paper mentions); use it when voids open, because by volume the rarefied gas of
a spall void receives nearly all of the defect and the time step collapses.  The paper gives no value of
:math:`\omega`.  On its titanium spallation case 1 (1030 MPa, share by mass) the lowest pressure is 0,
:math:`-0.36` and :math:`-0.67` GPa for instantaneous relaxation and :math:`\omega = 3.5` and
0.5 Pa\ :sup:`-1` s\ :sup:`-1`; the paper shows :math:`-0.38` GPa.  The value 3.5 is one fitted by
Biasiori-Poulanges & Schmidmayer (2023), not one given for this test.

Compaction energy and GTN yield function (:code:`elastic.compaction = 1`, default 0).  For one solid phase (phase 0)
with a fluid in its pores, the compaction energy :math:`\kappa = (\chi/\rho_r)\,f(\alpha_s)\,|\xi-1|^n` and the
Gurson--Tvergaard--Needleman yield function of Favrie & Schmidmayer (2026) are available.  The paper's
configuration variable is carried as a reference solid fraction :math:`\alpha_{ref} = \alpha_s/\xi` in the
:code:`cfun` field.  Inputs: :code:`elastic.chi0` or :code:`elastic.species.chi` (compaction modulus),
:code:`elastic.gtn0` or :code:`elastic.species.gtn` (1 = GTN, 0 = von Mises), :code:`elastic.comp.n` (2),
:code:`elastic.comp.nkappa` (0.5), :code:`elastic.comp.akappa` (1.8e5), :code:`elastic.gtn.q1/q2/q3` (1).

.. warning::

   Implemented and stable, **not validated**.  Three points are assumptions, because the paper does not settle
   them: the plastic flow rule (associated), a factor :math:`1/\rho_r` in :math:`\kappa`, and the unit of the GTN
   viscosity.  On the porous-titanium cases of the paper (0.01, 1 and 10 % air) the wave arrival times, the
   presence or absence of spall and the rear-face velocity peak agree with the published figures, but a control
   run without these terms gives the same result to three digits: with the paper's parameters they change almost
   nothing here, and the sustained negative pressures the paper reports are not reproduced.

The paper's two-dimensional tests run with the same options: an aluminium cylinder striking a 20 % porous titanium
plate (:code:`input_2D_AlCylinder_PorousTi`) and a titanium cylinder perforating a 40 % porous aluminium plate
(:code:`input_2D_TiCylinder_PorousAl`), both at 700 m/s on 400 x 400 cells.  The script
:code:`favrie_schmidmayer_2d_plots.py` draws them in the split layout of the paper's Figs. 12 and 13 (numerical
schlieren by its eq. 122, mixture density, aluminium volume fraction).  The published pictures are reproduced
feature by feature (compaction under the impact, the flattened projectile and the jet in the first; compaction,
bending and full perforation with a nearly undeformed projectile in the second, whose position at 20 microseconds
is 13.9 mm against about 13.7 mm in the paper's figure); the paper gives no numbers for these tests.

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

Elasto-viscoplasticity
======================

:code:`elastic.plastic = 1` (default 0) lets the solid yield.  The model is the plastic relaxation of
Favrie and Schmidmayer, *A multiphase compressible flow model including elasto-viscoplastic solids with
irreversible compaction*, Comput. Methods Appl. Mech. Engrg. 455 (2026) 118880: the elastic deformation relaxes
when the yield function is exceeded,

.. math::

   \frac{d\mathbf{F}_e^{-1}}{dt} + \mathbf{F}_e^{-1}\,\nabla\mathbf{u} = \mathbf{F}_e^{-1}\,\dot{\mathbf{R}},
   \qquad \dot{\mathbf{R}} = \dot\gamma\,\frac{\partial g}{\partial\boldsymbol\tau},

with the von Mises yield function :math:`g = \sigma_\text{eq} - \sigma_y`,
:math:`\partial g/\partial\boldsymbol\tau = 3\mathbf{S}/(2\sigma_\text{eq})` and the Perzyna rate
:math:`\dot\gamma = \langle g\rangle / K_y`.  It is applied once per step after the flow update, cell by cell:
the cobasis is moved back toward the yield surface along the flow direction,
:math:`\mathbf{E}\leftarrow\mathbf{E}\exp(\Delta\gamma\,\mathbf{N})` (volume preserving), with
:math:`\Delta\gamma` from a backward-Euler (radial-return) iteration on the non-linear
:math:`\sigma_\text{eq}(\Delta\gamma)`.  The elastic energy released stays in the conserved total energy and
becomes internal energy of the solid: plastic work heats the material.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Input
     - Default
     - Meaning
   * - :code:`elastic.plastic`
     - 0
     - 1 = von Mises / Perzyna plastic relaxation
   * - :code:`elastic.sigma_y0`, :code:`elastic.sigma_y1`
     - 0
     - yield stress of phase 0 / phase 1 (:math:`\le 0`: that phase never yields)
   * - :code:`elastic.Ky0`, :code:`elastic.Ky1`
     - 0
     - plastic viscosity :math:`K_y` [stress x time]; 0 = rate-independent (perfect plasticity)

Plot fields :code:`plastic_rate` (plastic strain rate of the last step) and :code:`plastic_seq` (von Mises stress).

**Verification.**  :code:`tests/FlowElasticSolid/input_1D_CuAlImpact` is test 5.1 / Fig. 4 of that paper (copper at
+400 m/s on aluminium at -200 m/s, 50 microseconds).  Against the curves of the published figure: pressure, normal
stress, velocity and density on both sides within 0.14 % with plasticity (plastic fronts at 0.3060 / 0.7755 against
0.3061 / 0.7755) and within 0.5 % without.  The elastic precursors sit at the Hugoniot elastic limit
:math:`\sigma_y (K + 4\mu/3)/(2\mu)` to 0.3 %.

**Not implemented from that paper:** the Gurson--Tvergaard--Needleman yield function, the compaction energy and
its variable, finite-rate pressure relaxation, and more than two materials.  Its titanium tests (two metals plus
air or water) cannot be run until Hydro2 carries more than two phases.

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
