.. _hydro2-phasechange:

==========================================================
:fas:`tint;fa-fw` Hydro2: Liquid--Vapour Phase Change
==========================================================

:code:`phasechange.on = 1` adds liquid <-> vapour mass transfer by **thermodynamic-equilibrium
(p-T-g) relaxation**, the standard closure for phase transition in the six-equation model
(Saurel, Petitpas & Abgrall, JFM 607, 2008; Pelanti & Shyue, JCP 259, 2014).
Default off; :code:`src/Integrator/Hydro2_PhaseChange.H`.

After the hydrodynamic step and the pressure relaxation, every cell that holds liquid is brought to the
state in which liquid and vapour have the same pressure, temperature and Gibbs free energy, at fixed
mass, volume and total energy.  **Nothing is tuned**: there is no rate constant and no mass-diffusion
coefficient.  The transfer is driven by the temperature, energy and pressure field the flow produces
(expansion -> cavitation / flashing, compression or cooling -> condensation).

.. note::

   This is separate from the older Spalding-number evaporation path (:code:`apply_vaporization`,
   :code:`Dv`, :code:`Lv`, :code:`Y_infinity`), which has not been removed.

.. contents::
   :local:
   :depth: 2


Model
=====

Each phase is a stiffened gas with energy and entropy constants :math:`q`, :math:`q'`
(Le Metayer, Massoni & Saurel, 2004):

.. math::

   e = \frac{p + \gamma\pi}{(\gamma - 1)\rho} + q, \qquad
   T = \frac{p + \pi}{(\gamma - 1)\,c_v\,\rho}, \qquad
   g = (\gamma c_v - q')\,T - c_v T \ln\!\frac{T^\gamma}{(p + \pi)^{\gamma - 1}} + q .

:math:`\gamma`, :math:`\pi`, :math:`c_v` are the phase's Hydro2 EOS (:code:`eos<k>.gamma`, :code:`eos<k>.p0`,
:code:`eos<k>.cv`); :math:`q` and :math:`q'` come from the material data file.

.. warning::

   Hydro2's plotted temperature :code:`T` uses the Tammann convention
   :math:`T = (p + \gamma\pi)/((\gamma - 1)\rho c_v)`.  For a gas (:math:`\pi = 0`) the two agree, for a liquid
   they do **not**.  Phase change uses the form above throughout: read :code:`pc_T`, not :code:`T`.

**Non-condensable gas.**  The gas phase is one fluid (air and vapour share one EOS), but only its *vapour*
part can condense, and it is the vapour **partial pressure** that enters the equilibrium.
The partial density of vapour in the gas phase, :math:`m_\text{vap} = (\alpha\rho)_\text{gas}\,Y_v`, is
transported as an extra conserved field (:code:`vapor`), and the saturation condition is

.. math::

   g_\text{liquid}(p, T) = g_\text{vapour}(x_v\,p,\, T), \qquad x_v = Y_v .

So air at room temperature does not condense, a droplet in air only humidifies its interface cells, and
a liquid whose pressure falls below :math:`p_\text{sat}(T)` cavitates into (nearly) pure vapour.

The cell solve is algebraic (nested bisection in :math:`p`, :math:`T` and :math:`m_\text{vap}`) on the mass,
volume, energy and Gibbs conditions.  The hydrodynamic phase energies of Hydro2 exclude :math:`q`, so the
conserved :math:`\rho E` changes by :math:`-(m_\text{vap}^\text{new} - m_\text{vap}^\text{old})(q_\text{vap} - q_\text{liq})`:
that is the latent heat.


Phase-change inputs
===================

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Input
     - Default
     - Meaning
   * - :code:`phasechange.on`
     - 0
     - 1 = phase change on.
   * - :code:`phasechange.liquid`
     - 1
     - Hydro2 phase index of the liquid (the other phase is the gas).
   * - :code:`phasechange.material_file`
     - :code:`data/hydro2_materials.dat`
     - Material data file, one material per line:
       :code:`name  phase  gamma  pi[Pa]  cv[J/kg/K]  q[J/kg]  q'[J/kg/K]`.
   * - :code:`phasechange.liquid_material`
     - empty
     - Entry of the file used for the liquid.
   * - :code:`phasechange.vapor_material`
     - empty
     - Entry of the file used for the vapour.
   * - :code:`phasechange.q_liquid`, :code:`phasechange.qp_liquid`
     - from the file
     - Direct override of the liquid's :math:`q`, :math:`q'`.
   * - :code:`phasechange.q_vapor`, :code:`phasechange.qp_vapor`
     - from the file
     - Direct override of the vapour's :math:`q`, :math:`q'`.
   * - :code:`phasechange.Yv0`
     - 0.0
     - Initial vapour mass fraction of the gas (0 dry, 1 pure vapour).
   * - :code:`phasechange.alpha_min`
     - 1e-9
     - Cells with less liquid or gas volume than this are skipped.
   * - :code:`phasechange.T_lo`, :code:`phasechange.T_hi`
     - 50, 5000
     - Temperature bracket of the solves.
   * - :code:`phasechange.evaporate`
     - 1
     - 0 disables liquid -> vapour.
   * - :code:`phasechange.condense`
     - 1
     - 0 disables vapour -> liquid.

The :math:`\gamma`, :math:`\pi`, :math:`c_v` of a material in the file **must match the deck's EOS** to 1 %,
otherwise the run aborts at :code:`Parse`.  At start-up the solver prints :math:`T_\text{sat}` at 1 atm and
the latent heat there -- check them.

.. code-block:: none

   eos0.gamma = 1.43          # phase 0: vapour
   eos0.p0    = 0.0
   eos0.cv    = 1040.0
   eos1.gamma = 2.35          # phase 1: liquid water
   eos1.p0    = 1.0e9
   eos1.cv    = 1816.0

   phasechange.on = 1
   phasechange.liquid = 1                       # phase 1 (eta = 0) is the liquid; phase 0 the gas
   phasechange.liquid_material = water_liquid
   phasechange.vapor_material  = water_vapor
   phasechange.material_file   = ../data/hydro2_materials.dat
   phasechange.Yv0 = 1.0                        # the gas is pure vapour

.. note::

   The default :code:`material_file` path is relative to the working directory.  The test decks are run
   from :code:`bin/`, hence :code:`../data/hydro2_materials.dat`.

Materials shipped in :code:`data/hydro2_materials.dat`:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Name
     - Note
   * - :code:`water_liquid`, :code:`water_vapor`
     - Le Metayer et al. (2004) fit; :math:`T_\text{sat}`\ (1 atm) = 373.3 K, latent heat 2.16e6 J/kg.
       Use this pair when the cavitation threshold matters.
   * - :code:`steam_as_air`
     - "Steam with the properties of air" (:math:`\gamma` 1.4, :math:`c_v` 717.86); :math:`q`, :math:`q'` fitted
       against :code:`water_liquid` at 1 atm only.
   * - :code:`water_tammann`, :code:`steam_as_air_tammann`
     - Water as the Tammann liquid of the shock-droplet decks (:math:`\gamma` 7.15, :math:`\pi` 3e8) and its
       partner, fitted at 1 atm only.


Verification
============

Decks in :code:`tests/FlowPhaseChange`, checks in :code:`tests/FlowPhaseChange/reference`.

.. list-table::
   :header-rows: 1
   :widths: 24 26 50

   * - Test
     - Deck
     - Result
   * - 0D equilibrium
     - :code:`input_0D_Equilibrium`
     - Four initial states against an independent scipy solution: p, T, vapour and liquid mass, :math:`\alpha`
       all to the printed digits; mass exact, energy 2e-16.
   * - Cavitation tube
     - :code:`input_1D_CavitationTube`
     - Centre pressure 51042 Pa = :math:`p_\text{sat}(T)` to 2e-9 (5401 Pa without mass transfer); vapour
       fraction 0.01 -> 0.136.
   * - Cavitation bubble
     - :code:`input_2D_CavitationBubble`
     - :math:`|p/p_\text{sat}(T) - 1|` in the cavity 2e-11 at every output; exactly symmetric under x <-> y.
   * - Advected saturated mixture
     - :code:`input_0D_Equilibrium` + velocity
     - Equal to the 0D reference to the printed digits, mass drift 0.
   * - Shock--droplet atomization
     - :code:`FlowShockDroplet/Atomization/input_2mm_Ma3`
     - Runs to t* = 2.08 with and without phase change; drop motion within 5 % of the no-phase-change run.

:code:`reference/saturation_table.py` compares the material data with IAPWS (no solver):
:code:`water_liquid` + :code:`water_vapor` :math:`p_\text{sat}` within +-3.6 % for 300--400 K.


Known limitations
=================

* **Steam-as-air pairs are exact at 373.15 K only.**  Air's gas constant fixes the wrong
  Clausius--Clapeyron slope: :math:`p_\text{sat}` is -83 to -87 % at 300 K and +66 to +70 % at 400 K.  On the
  atomization deck the material's saturation pressure at 300 K is 585 Pa (real water: 3.5 kPa).
* **Cost.**  :code:`phasechange.on = 1` costs about 2.7x the flow solver on the atomization deck: one p-T
  solve and Gibbs difference in *every* cell on every ghost fill, since every cell holds a trace of the
  other phase.
* **The plot field T is wrong in the liquid** when :code:`phasechange.on = 0` (Tammann convention, see above).
* **Grid imprint.**  The 2D cavity is diamond shaped (clearest in T).
* With phase change the atomizing drop shows a dimple on the windward axis at 80 us that is not in the
  reference; cause not investigated.
* Not done: AMR reflux of the vapour field, a PelePhysics hook.
