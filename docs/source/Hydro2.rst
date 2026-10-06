.. _hydro2:

==============================================
:fas:`fire;fa-fw` Hydro2 (6-equation) Solver
==============================================

:code:`Integrator::Hydro2` is the compressible two-phase solver of FLAMES.
It is built on the six-equation diffuse-interface model of Saurel, Petitpas & Berry (2009)
as re-cast for bubble dynamics by Schmidmayer, Bryngelson & Colonius (2020), and it is run
through the stand-alone :code:`hydro2` executable (:code:`bin/hydro2-2d-g++`, :code:`bin/hydro2-3d-g++`).

These pages are the hand-written part of the Hydro2 documentation: what the model carries,
what every solid / boundary / shell input does, and how to set up and check a run.
The complete, auto-generated list of inputs still lives in :ref:`inputs`; the regression decks are
listed in :ref:`tests`.
Everything documented here was read out of :code:`src/Integrator/Hydro2.cpp` (the :code:`Parse` section),
the solid headers next to it, and the header comments of the decks in :code:`tests/Flow*`.

.. contents::
   :local:
   :depth: 2

What the model carries
======================

The distinguishing feature of the six-equation model is what it refuses to assume: the two phases
are *not* forced into pressure equilibrium during the transport step.
Each phase carries its own pressure and internal energy through the hyperbolic update, and
mechanical equilibrium is restored afterwards by a stiff, cell-local pressure relaxation.

The state vector is

.. math::

   \mathbf{U} = \begin{bmatrix}
   \eta & \alpha_1\rho_1 & \alpha_2\rho_2 & \rho\mathbf{u} & \rho E & E_1 & E_2
   \end{bmatrix}^T ,

with :math:`\eta \equiv \alpha_1` the volume fraction of phase 0 (so :math:`\alpha_2 = 1 - \eta`),
two partial densities, one mixture momentum, the redundant mixture total energy :math:`\rho E`
(carried so that the scheme stays conservative across a shock) and the two phasic internal energies
:math:`E_k = \alpha_k \rho_k e_k`.
The mixture relations are

.. math::

   \rho = \alpha_1\rho_1 + \alpha_2\rho_2, \qquad
   p = \alpha_1 p_1 + \alpha_2 p_2, \qquad
   \rho E = E_1 + E_2 + \tfrac{1}{2}\rho|\mathbf{u}|^2 .

Each phase is closed by a stiffened-gas (Tammann) equation of state,

.. math::

   p_k = (\gamma_k - 1)\rho_k e_k - \gamma_k \pi_k, \qquad
   T_k = \frac{p_k + \gamma_k \pi_k}{(\gamma_k - 1)\rho_k c_{v,k}}, \qquad
   a_k = \sqrt{\gamma_k (p_k + \pi_k)/\rho_k},

where :math:`\pi_k` is the input :code:`eos<k>.p0` (:math:`\pi = 0` recovers a calorically perfect gas).

.. note::

   **Phase numbering.** The inputs are numbered from zero: *phase 0* is the phase with
   :math:`\eta = 1` (:code:`eos0`, :code:`mu0`, :code:`density0.ic`, plot field :code:`rho_eta0`),
   *phase 1* is the phase with :math:`\eta = 0`.
   The theory (and the dissertation) number the same phases 1 and 2.

On top of the two-fluid core the integrator has a set of optional physics packages, each off by default:

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Feature
     - Switch
     - Page
   * - Non-deforming embedded solid (diffuse or sharp wall), static or moving
     - :code:`apply_embedded_solid`, :code:`solid.*`
     - :ref:`hydro2-solids`
   * - Deforming hyperelastic solid phase
     - :code:`elastic.on`, :code:`elastic.*`
     - :ref:`hydro2-elastic`
   * - Ablating (receding) embedded solid, in-depth conduction
     - :code:`ablation.on`, :code:`ablation.*`
     - :ref:`hydro2-ablation`
   * - Liquid--vapour phase change by thermodynamic relaxation
     - :code:`phasechange.on`, :code:`phasechange.*`
     - :ref:`hydro2-phasechange`
   * - Visco-elastic (polymer / lipid) shell on the interface
     - :code:`marmottant`, :code:`shell.*`
     - :ref:`hydro2-shell`
   * - Domain boundary conditions (Neumann, Dirichlet, reflective, periodic, NSCBC)
     - :code:`<field>.bc.*`, :code:`nscbc.*`
     - :ref:`hydro2-bc`
   * - Building, running, checking, choosing numerics
     -
     - :ref:`hydro2-howto`


Anatomy of a deck
=================

Hydro2 inputs carry **no prefix**: the :code:`hydro2` executable reads them from the top level of the
input file.  A deck is organised in the same blocks every time.

.. code-block:: none

   ### OUTPUT ###
   plot_file = ./tests/MyCase/output
   amr.plot_dt = 0.25

   ### MESHING / DIMENSIONS ###
   amr.n_cell           = 384 240
   amr.max_level        = 2
   amr.blocking_factor  = 4
   geometry.prob_lo     = -10.0 -15.0 0.0
   geometry.prob_hi     =  30.0  15.0 0.0
   geometry.is_periodic = 0 0 0

   ### TIME STEPPING ###
   timestep            = 1e-5
   stop_time           = 60.0
   dynamictimestep.on  = 1
   dynamictimestep.max = 5e-3
   dynamictimestep.min = 1e-9
   cfl                 = 0.4
   nghost              = 4
   integration.type    = RungeKutta
   integration.rk.type = 3

   ### EOS + TRANSPORT (one block per phase) ###
   eos0.gamma = 1.4
   eos0.p0    = 0.0
   eos0.cp    = 1005.0
   eos0.cv    = 717.86
   mu0        = 2.5
   # ... eos1.*, mu1

   ### ETA + PER-PHASE INITIAL CONDITIONS ###
   eta.ic.type = constant
   eta.ic.constant.value = 1.0
   density0.ic.type  = constant
   velocity0.ic.type = expression
   pressure0.ic.type = constant
   # ... density1 / velocity1 / pressure1

   ### INTERACTIONS ###
   epsilon = 0.1
   sigma   = 0.0

   ### BOUNDARY CONDITIONS ###          (see the Boundary conditions page)
   ### REFINEMENT CRITERIA ###          (see the How-to page)

   ### SOLVER ###
   Riemann_Solver.type = hllc
   Limiter.type        = minmod

Required inputs
---------------

These have no default; the run aborts at :code:`Parse` without them.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Input
     - Meaning
   * - :code:`cfl`
     - CFL number of the acoustic (convective) time-step limit.
   * - :code:`mu0`, :code:`mu1`
     - Dynamic (linear) viscosity of each phase.  :code:`0` = inviscid.
   * - :code:`epsilon`
     - Diffuse-interface thickness.  Must be positive.
   * - :code:`eos<k>.gamma`, :code:`eos<k>.p0`, :code:`eos<k>.cv`, :code:`eos<k>.cp`
     - Stiffened-gas constants of phase :code:`k` = 0, 1.
   * - :code:`timestep`, :code:`stop_time`
     - Nominal level-0 time step and end time (base integrator).

Common optional inputs
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Input
     - Default
     - Meaning
   * - :code:`cfl_v`
     - :code:`cfl`
     - CFL number of the viscous / conduction / capillary / body-force limits.
   * - :code:`nghost`
     - 2
     - Number of ghost cells.  NSCBC needs 2 (2D only) or 4; :code:`weno5` needs at least 3 (use 4).
   * - :code:`mu0_b`, :code:`mu1_b`
     - 0
     - Bulk viscosity of each phase.
   * - :code:`thermal.conduction`
     - 0
     - 1 = Fourier heat conduction, :math:`k = \mu(\eta)\,c_p(\eta)/\mathrm{Pr}`.
   * - :code:`thermal.Pr`
     - 0.72
     - Prandtl number.
   * - :code:`sigma`
     - 0
     - Surface tension coefficient (reused as the bare-interface tension when :code:`marmottant = 1`).
   * - :code:`apply_surface_tension`
     - false
     - Apply the capillary source.
   * - :code:`apply_weight`
     - false
     - Apply the weight (gravity) source; :code:`grav` (default 9.81) is the acceleration.
   * - :code:`pref`
     - 0
     - Reference pressure (Roe solver).
   * - :code:`small`
     - 1e-8
     - Small regularisation value.
   * - :code:`cutoff`
     - 1e-8
     - :math:`\eta` cutoff value.
   * - :code:`equalize_ic_pressure`
     - 0
     - Equalise the phasic pressures of the initial condition to avoid a spurious start-up oscillation.
   * - :code:`plot_minimal`
     - 0
     - 1 = drop the pure diagnostic fields from the plotfiles (roughly 70 % smaller in 3D).
   * - :code:`nan_check`
     - 0
     - 1 = per-cell NaN/Inf guard (slow; debugging only).
   * - :code:`verbose`, :code:`relax_diag`
     - 0
     - Per-call informational messages / per-stage relaxation diagnostics.

Initial conditions are selected with :code:`<name>.ic.type` (:code:`constant` or :code:`expression`;
:code:`eta.ic` also accepts :code:`laminate`, :code:`bmp` and :code:`png`).
The fields are :code:`eta.ic`, :code:`density<k>.ic`, :code:`velocity<k>.ic`, :code:`pressure<k>.ic`
and the diffuse-boundary sources :code:`m0.ic`, :code:`u0.ic`, :code:`q.ic`.

Plot fields
-----------

The plotfile always holds :code:`eta`, :code:`rho_eta0`, :code:`rho_eta1`, :code:`density`,
:code:`momentum`, :code:`velocity`, :code:`pressure`, :code:`energy0`, :code:`energy1`,
:code:`energy_per_vol`, :code:`T`, :code:`UE_per_vol`, :code:`KE_per_vol`, :code:`shell`,
:code:`Gamma`, :code:`grad_eta` and :code:`kappa`.
Each optional package adds its own:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Extra plot fields
   * - Embedded solid
     - :code:`phi`, :code:`grad_phi`
   * - Ablation
     - :code:`ablation_q`, :code:`ablation_v`, and :code:`ablation_Ts` with in-depth conduction
   * - Elastic solid
     - :code:`ebasis` (cobasis components :code:`1x` ... :code:`3z`),
       :code:`elastic_Sxx` ... :code:`elastic_Syz`, :code:`elastic_W`
   * - Phase change
     - :code:`vapor`, :code:`pc_T`, :code:`pc_Tsat`, :code:`pc_Yv`, :code:`pc_dm`


Feature pages
=============

.. toctree::
   :maxdepth: 2

   Hydro2_Solids
   Hydro2_ElasticSolid
   Hydro2_Ablation
   Hydro2_PhaseChange
   Hydro2_BoundaryConditions
   Hydro2_Shell
   Hydro2_HowTo
