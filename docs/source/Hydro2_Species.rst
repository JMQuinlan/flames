.. _hydro2-species:

Species inside the two phases
=============================

Default **off** (:code:`species.on = 0`).  What exists today is **stage 1: passive transport**.  The equation of
state, the relaxation and the flow do not see the species yet -- with :code:`species.on = 1` the flow is
bit-identical -- and chemistry, species thermodynamics and phase change on species are later stages.  The plan
and the reasoning are in :code:`docs/design/Hydro2_MultiSpecies_Chemistry_Design.md`.

Species model
-------------

Hydro2 keeps its two phases: one volume fraction :math:`\eta` and the phase partial densities
:math:`\alpha_0\rho_0`, :math:`\alpha_1\rho_1`.  Each phase is a *mixture of species*.  A substance present
as a liquid and as a vapour is two entries, one in each phase.  The transported field is the species partial
density

.. math::

   m_i = \alpha_k\,\rho_k\,Y_{k,i}, \qquad \sum_{i\in k} m_i = \alpha_k\rho_k, \qquad
   \frac{\partial m_i}{\partial t} + \nabla\cdot\big(F_k\,Y_{k,i}\big) = 0,

with :math:`F_k` the mass flux of phase :math:`k` returned by the Riemann solver and :math:`Y` taken from the
upwind cell.

**Closure.**  The small fractions are advected first and the *largest* species of the phase takes the
remainder, so a trace species is never lost to a normalisation:

* on a face, every species but the majority one of the upwind cell gets :math:`F_k Y_i`, accumulated from the
  smallest up; the majority species gets :math:`F_k - \sum_\text{others}`.  The species fluxes add up to the
  phase mass flux exactly;
* in a cell, after every Runge--Kutta stage, the majority species is reset to
  :math:`\alpha_k\rho_k - \sum_\text{others}`.

Species inputs
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Input
     - Default
     - Meaning
   * - :code:`species.on`
     - 0
     - 1 = transport species
   * - :code:`species.phase0.names`
     - (none)
     - names of the species of phase 0 (the :math:`\eta = 1` phase)
   * - :code:`species.phase1.names`
     - (none)
     - names of the species of phase 1
   * - :code:`species.ic.type`, :code:`species.ic.expression.regionN`
     - required
     - mass fractions *within their phase*, one component per species, phase 0 names first.  Give every species;
       the largest value of a cell is the one closed to the phase mass
   * - :code:`species.bc.type.xlo` ...
     - zero gradient
     - boundary condition of the partial densities.  **Required** (periodic) on a periodic domain

Plot fields: :code:`species_<name>` (partial density) and :code:`Y_<name>` (mass fraction within the phase).

.. code-block:: none

   species.on = 1
   species.phase0.names = L1 L2
   species.phase1.names = G1 G2 Vap
   species.ic.type = expression
   species.ic.expression.region0 = "0.75"
   species.ic.expression.region1 = "0.25"
   species.ic.expression.region2 = "0.8 - 1.0e-6"
   species.ic.expression.region3 = "0.2"
   species.ic.expression.region4 = "1.0e-6"

Species verification
--------------------

Decks in :code:`tests/FlowSpecies`, check :code:`tests/FlowSpecies/reference/species_check.py`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Deck
     - Result
   * - :code:`input_1D_Advection`
     - five species in two phases, one turn of a periodic box: sum of species equals the phase density to
       :math:`9\times10^{-15}`, species masses constant to :math:`6\times10^{-14}`, no negative values, smooth
       profiles first order (donor cell), a uniform :math:`10^{-10}` trace unchanged
   * - :code:`input_1D_TwoPhaseSlab`
     - liquid slab with two species in a gas with three (one of them the vapour of a liquid species): the closure
       holds in every cell including the interface; uniform pressure and velocity kept
   * - :code:`input_1D_SodSpecies`
     - driver and driven gas as species: the trace is unchanged to 12 digits through shock and rarefaction;
       driver gas past the diaphragm -0.64 / -0.30 / -0.13 % at 200 / 400 / 800 cells

In all three the flow fields are bit-identical to the run with :code:`species.on = 0`.

Chemistry (native one-step backend)
-----------------------------------

:code:`chem.on = 1` (default 0; needs :code:`species.on = 1`) adds one irreversible reaction inside one phase,

.. math::

   \text{reactant} \rightarrow \text{product} + Q, \qquad \frac{dY_r}{dt} = -A\,Y_r\,\exp(-T_a/T),

operator split once per step at constant volume, with Runge--Kutta sub-steps limited by the relative change of
:math:`Y` and :math:`T`.  The phase energies stay sensible; the heat of reaction is booked into the phase energy
and the total energy when the reaction happens.  Reactant and product share the equation of state of their phase.
It exists to verify transport, energy bookkeeping and splitting against closed forms before a library supplies real
mechanisms through the same hook.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Input
     - Default
     - Meaning
   * - :code:`chem.on`
     - 0
     - 1 = react
   * - :code:`chem.reactant`, :code:`chem.product`
     - required
     - species names (same phase)
   * - :code:`chem.A`
     - required
     - pre-exponential factor [1/s]
   * - :code:`chem.Ta`
     - 0
     - activation temperature :math:`E_a/R`
   * - :code:`chem.Q`
     - required
     - heat released per unit mass of reactant
   * - :code:`chem.T_min`
     - 0
     - no reaction below this temperature

Checks (:code:`tests/FlowChemistry`): the constant-volume reactor reproduces :math:`Y = e^{-At}` to
:math:`2\times10^{-12}` and a thermal explosion to :math:`10^{-6}`; a centre-ignited Chapman--Jouguet
detonation (:code:`input_1D_Detonation`, :math:`\gamma = 1.2`, :math:`Q = 50`, :math:`T_a = 20`) travels at the
Chapman--Jouguet speed to :math:`+0.3\,\%` (identical at 5 and 10 cells per half-reaction length), reaches the
von Neumann pressure to :math:`-1.1\,\%`, follows the ZND profile within :math:`2.4\,\%` and ends at the Taylor-wave
pressure to :math:`+1.0\,\%`; total energy (with the heat released) is kept to :math:`6\times10^{-6}` and mass to
round-off.

.. note::

   Run detonations with :code:`Riemann_Solver.type = hlle`.  On a thin periodic strip the HLLC solver develops a
   transverse (odd--even) instability behind a strong grid-aligned shock; HLLE does not.

Species-dependent equation of state (stage 2a)
----------------------------------------------

With :code:`species.eos = mixture` (default :code:`fixed`: the flow does not see the species) every species of a
phase is a stiffened gas with its own :math:`\gamma_i` and :math:`c_{v,i}` and the stiffness :math:`\pi_k` of its
phase.  The species of a phase share pressure and temperature, so the phase is itself a stiffened gas with

.. math::

   c_{v} = \sum_i Y_i\,c_{v,i}, \qquad \gamma = \frac{\sum_i Y_i\,\gamma_i\,c_{v,i}}{c_v}, \qquad \pi = \pi_k ,

evaluated in every cell from the species partial densities and used wherever the solver read the phase constants
(Riemann states, phase-energy source, relaxation, temperature, sound speed, time step, chemistry).

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Input
     - Default
     - Meaning
   * - :code:`species.eos`
     - fixed
     - :code:`mixture` = species-dependent :math:`\gamma`, :math:`c_v`
   * - :code:`species.phase0.gamma`, :code:`species.phase0.cv`
     - required with mixture
     - one value per species of phase 0 (same for :code:`phase1`)

Checks (:code:`tests/FlowSpecies`, script :code:`two_gamma_check.py`): with the option off the flow is bit-identical;
with identical species it reproduces the fixed result to round-off; a uniform mixture has the closed-form
:math:`\gamma`, :math:`T` and energy; the shock tube between two gases (:math:`\gamma = 1.4\,|\,5/3`,
:code:`input_1D_TwoGammaShockTube`) converges to the exact two-gas Riemann solution like the same problem run with the
gases as two phases; a reactor whose product has a different :math:`\gamma` follows the closed form to :math:`10^{-6}`.

.. warning::

   An advected contact between two species of different :math:`\gamma`
   (:code:`input_1D_IsobaricSpeciesContact`) does **not** keep the pressure uniform: 5 % error at 100 cells, 3 % at
   800.  This is the known behaviour of a conservative species scheme.  Two *phases* keep the pressure uniform.
   Use phases for a gas--gas interface that must stay quiet, species for mixtures and reaction products.

Not covered yet: NSCBC boundaries, AMR levels, species-dependent conductivity.

Several solids in one phase (stage 2b)
--------------------------------------

Species of one phase may also have different stiffness.  Give, for every species of that phase,
:code:`species.phaseK.pinf` and :code:`species.phaseK.rho_ref`; the phase constants of a cell then follow the
volume-weighted closure of the five-equation models with the volume fractions taken at the reference densities,

.. math::

   z_i = \frac{Y_i/\rho_{ref,i}}{\sum_j Y_j/\rho_{ref,j}}, \qquad
   \frac{1}{\gamma-1} = \sum_i \frac{z_i}{\gamma_i-1}, \qquad
   \frac{\gamma\pi}{\gamma-1} = \sum_i \frac{z_i\gamma_i\pi_i}{\gamma_i-1} .

This is exact where one species fills the phase and a closure in the mixed cells of a solid--solid contact.  With
:code:`elastic.species.mu`, :code:`elastic.species.sigma_y` and :code:`elastic.species.Ky` the elastic solid takes
its shear modulus, yield stress and plastic viscosity per species as well.  This is how three materials (two metals
and air) run in the two-phase solver: :code:`input_1D_CuAlImpact_Species` reproduces the published copper--aluminium
impact to 0.3 %, and :code:`input_1D_TiSpallation` is a first comparison with the titanium spallation of
Favrie & Schmidmayer (2026) (loading waves within 3 %, same spall plane; fragment pattern differs -- see the
elastic-solid page).

.. warning::

   A contact between two species of different stiffness is not energy-consistent: expect a spike in the cell at
   the contact (up to +34 % in pressure at the moment of impact in the spallation test, decaying afterwards).

Second-order species transport
------------------------------

:code:`species.order = 2` (default 1) replaces the upwind-cell mass fraction at a face by a van Leer limited
linear value.  On the advection test the error drops by a factor 7 to 13.  It is **required** for detonations with
chain-branching chemistry: at first order radicals leak forward through the captured front (e-folding length about
one cell) and the induction zone disappears, at any resolution.

Thermally perfect species and finite-rate chemistry (PelePhysics mechanism)
---------------------------------------------------------------------------

A binary built with a PelePhysics mechanism directory,

.. code-block:: bash

   make -j 14 POSTFIX=2d-g++-bd AMREX_TARGET=ext/AMReX-Codes/amrex/2d-g++-26.02 \
        HYDRO2_MECH=ext/AMReX-Combustion/PelePhysics/Mechanisms/BurkeDryer bin/hydro2-2d-g++-bd

accepts :code:`species.eos = pele` and :code:`chem.backend = pele`.  Only the mechanism's generated
:code:`mechanism.H` / :code:`mechanism.cpp` are used (NASA-polynomial thermodynamics, molecular weights, production
rates); the constant-volume reactor is an adaptive Runge--Kutta 4(5) of our own, no SUNDIALS.  The species of one
phase must be the mechanism's species in its order, and the deck must be in SI units.  The phase energy remains a
positive sensible energy; the solver receives, cell by cell, the temperature from that energy and an effective
:math:`\gamma` and :math:`c_v` for which pressure and temperature are exact.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Input
     - Default
     - Meaning
   * - :code:`species.eos = pele`
     -
     - thermally perfect species of the mechanism
   * - :code:`chem.backend`
     - native
     - :code:`pele` = mechanism of the build
   * - :code:`chem.T_min`
     - 0
     - no reaction below this temperature [K]
   * - :code:`chem.rtol`, :code:`chem.atol`
     - 1e-6, 1e-12
     - tolerances of the reactor on the mass fractions
   * - :code:`chem.max_sub`
     - 20000
     - sub-step limit per cell and step (a warning is printed when reached)

Checks with the Burke--Dryer H\ :sub:`2`--O\ :sub:`2` mechanism against Cantera (same mechanism,
:code:`h2o2_check.py`): constant-volume ignition of 2H\ :sub:`2` + O\ :sub:`2` + 7Ar at 1200 K, 1 atm follows
Cantera to :math:`6\times10^{-5}` in temperature and pressure; in the detonation at 6.67 kPa thermal plus chemical
energy is conserved to :math:`10^{-10}` and the reaction zone agrees with the steady ZND solution at the wave's own
speed (pressure within 4 %, temperature within 2.5 %).

Started from its ZND profile (:code:`input_1D_H2O2Detonation_ZND`, the way unsteady detonation runs are started in
the literature, e.g. Barwey et al. 2024 for this mixture) the detonation travels at 1615.8 m/s over 0.82 m, 0.05 %
below the Chapman--Jouguet speed 1616.65 m/s, with the von Neumann pressure at the front to 0.06 %.
With the HLLE solver the speed stays within 0.03 % of Chapman--Jouguet from 10 to 78 cells per induction length;
the peak pressure overshoots the von Neumann value by 0.2, 0.3, 0.5 and 0.3 % at 10, 20, 39 and 78 cells (a maximum
at intermediate resolution, as Barwey et al. report).  HLLC alone is not usable on this two-row strip: from 18 microseconds on the two rows diverge (the carbuncle-type
instability, see :code:`shock_fix.on`); use HLLE, or HLLC with the shock switch (speed :math:`-0.07\,\%`, peak pressure
:math:`+0.8\,\%` at 20 cells per induction length, rows identical).

.. note::

   Start detonations from the ZND profile (:code:`h2o2_check.py table`, initial condition type
   :code:`profile1d`).  After a blast or piston start this low-pressure mixture stays several per cent off the
   Chapman--Jouguet speed for more than a millisecond: acoustic signals from the recombination zone need 0.5 ms to
   reach the front from 150 mm behind it.  :code:`species.pele_sound = frozen` gives the solver the exact frozen
   sound speed (the default effective-:math:`\gamma` fit is 1.3--1.7 % high in burnt gas); it does not change the
   front history.

Species limitations
-------------------

* The species do not enter the equation of state yet (one equation of state per phase); the only chemistry is the
  one-step reaction above; no phase change on species.
* Single level: the species fluxes are not refluxed.
* First-order (donor-cell) face values, like the volume-fraction row.
