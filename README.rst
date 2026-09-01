.. raw:: html

   <p align="center">
       <img src="https://raw.githubusercontent.com/JMQuinlan/flames/development/docs/source/flames.png" alt="flames" width="480">
   </p>

   <p align="center">
       <a href="https://github.com/JMQuinlan/flames/tree/development"><img src="https://img.shields.io/github/last-commit/JMQuinlan/flames/development.svg?label=last%20commit%20%28development%29"></a>
       <a href="https://github.com/JMQuinlan/flames/pulls"><img src="https://img.shields.io/github/issues-pr/JMQuinlan/flames.svg"></a>
       <a href="https://github.com/JMQuinlan/flames/issues"><img src="https://img.shields.io/github/issues/JMQuinlan/flames.svg"></a>
       <a href="https://github.com/solidsgroup/alamo"><img src="https://img.shields.io/badge/fork%20of-solidsgroup%2Falamo-1f77b4"></a>
   </p>

.. getting-started:

FLAMES
======

**FLAMES** — di\ **F**\ fuse-interface para\ **L**\ lel **A**\ daptive **M**\ esh
navi\ **E**\ r-stokes **S**\ olver — provides explicit and implicit solvers for
**multiphase reacting flow** with block-structured adaptive mesh refinement,
built on `AMReX <https://amrex-codes.github.io/amrex/>`_ and
`Alamo <https://github.com/solidsgroup/alamo>`_.

FLAMES is a fork of Alamo. Everything Alamo does — phase field microstructure
evolution, strong-form solid mechanics, fracture, solid rocket propellant burn,
and fluid–solid interaction — is still here and still works. What FLAMES adds is
a **compressible fluid–fluid two-phase solver with phase change**: a diffuse
interface separating a gas and a liquid, each carried by its own equation of
state, with mass, momentum, and energy exchanged across the interface so that
evaporation, condensation, atomization, and burning can be simulated directly.

The target problems are the ones the logo is drawn from: a liquid body caught in
a high-speed compressible flow, breaking into progressively smaller fragments
that vaporize and burn, with the mesh refining to follow each fragment.

What FLAMES adds
----------------

* **A fluid–fluid diffuse-boundary integrator** (:code:`Integrator::Hydro2`).
  Where Alamo's :code:`Integrator::Hydro` treats a solid–fluid interface, FLAMES
  applies the same diffuse-boundary formalism to a **fluid–fluid** interface,
  with the two phases distinguished by their equation of state and blended
  through a phase field η.

* **Per-phase equations of state.** A calorically perfect gas (CPG) closure for
  the gaseous phase and a stiffened-gas (Tammann) closure for the liquid phase.
  The diffuse-boundary source terms themselves are EOS-agnostic by design.

* **Cahn–Hilliard phase-field evolution.** FLAMES evolves the interface with
  Cahn–Hilliard rather than the Allen–Cahn used in the underlying
  diffuse-boundary papers. Cahn–Hilliard is conservative
  (∫η is preserved), which is what makes the phase evolution
  mass-consistent with the mass/momentum/energy system it is coupled to.

* **Phase change.** Mass, momentum, and energy transfer across the interface
  from evaporation, condensation, and Stefan-type phase change, with the
  transfer rate ṁ closed either by kinetic theory or by a
  Cahn–Hilliard chemical-potential difference. Action–reaction constraints are
  enforced discretely, so what one phase loses the other gains exactly.

* **A family of approximate Riemann solvers** for the compressible fluxes, under
  :code:`src/Solver/Local/FluidRiemann/` — HLL, HLLE, HLLC and a number of
  low-Mach / all-Mach and higher-order variants, plus Roe, Lax–Friedrichs and
  upwind, selectable from the input file.

* **Characteristic boundary conditions.** Navier–Stokes characteristic boundary
  conditions (NSCBC, including a ghost-cell formulation) for non-reflecting
  inflow and outflow of acoustic waves, in :code:`src/BC/`.

* **Verification and regression cases** under :code:`tests/`, including 1D
  Stefan / saturation / conduction problems (:code:`tests/StefanVap`),
  vaporizing droplets and d²-law cases
  (:code:`tests/FlowVaporization`), shock–droplet interaction
  (:code:`tests/FlowShockDroplet`), and the Riemann and flow benchmarks
  inherited from Alamo (:code:`tests/Flow*`).

.. NOTE::
    :code:`Integrator::Hydro2` currently builds and runs in **2D only**.
    The rest of Alamo continues to build in both 2D and 3D.

Theoretical basis
-----------------

The diffuse-boundary framework FLAMES extends comes from two papers:

* Schmidt, E. M., Quinlan, J. M., and Runnels, B., *Self-similar Diffuse Boundary
  Method for Phase Boundary Driven Flow*, **Physics of Fluids**, Vol. 34, No. 11,
  2022. Inviscid Euler formulation with arbitrary essential/natural flux boundary
  conditions imposed on a diffuse interface; source terms distributed over the
  diffuse region recover the sharp-interface boundary conditions as ε → 0.

* Boyd, E. (née Schmidt), Sandall, E., Meier, M., Quinlan, J. M., and Runnels, B.,
  *A diffuse boundary method for phase boundaries in viscous compressible flow*,
  **Journal of Computational Physics**, Vol. 559, No. 114898, 2026. Extends the
  above to Navier–Stokes, including the angular momentum flux needed to impose
  no-slip on a diffuse boundary.

Both papers pose the method at a **solid–fluid** interface. FLAMES applies the
same machinery to a **fluid–fluid** interface with full per-phase transport and
phase change. For background on diffuse-interface methods generally, see
D. M. Anderson, G. B. McFadden, A. A. Wheeler, *Diffuse-interface methods in
fluid mechanics*, **Annual Review of Fluid Mechanics** 30:139–165 (1998).

Relationship to Alamo
---------------------

FLAMES tracks Alamo upstream and does not duplicate its documentation. For
anything not specific to the multiphase solver — the input-file syntax, the
:code:`Integrator` and :code:`Model` class hierarchies, the parser and
autodoc system, adding a new integrator, the numerics of the phase-field and
mechanics solvers, install scripts for your platform — go to the Alamo
documentation, which applies unchanged to FLAMES:

`Alamo documentation <https://solidsgroup.github.io/alamo/docs/>`_ ·
`Alamo repository <https://github.com/solidsgroup/alamo>`_ ·
`Alamo regression tests <https://solidsgroup.github.io/alamo/docs/Tests.html>`_

Downloading FLAMES
------------------

Download FLAMES using git:

.. code-block::

    git clone git@github.com:JMQuinlan/flames.git

If you do not have a Github account and/or you have not uploaded your public SSH
key, this will probably throw an error. You can download FLAMES using HTTPS
instead,

.. code-block::

    git clone https://github.com/JMQuinlan/flames.git

Note that you will not be able to push anything using HTTPS authentication.

The :code:`development` branch is the default and is where the stable multiphase
work lives. Feature branches carry the model variant in their name
(:code:`Hydro2_5Eqn_*`, :code:`Hydro2_6Eqn_*`, :code:`Hydro2_7Eqn_*`,
:code:`two-full-states`, and so on); these differ from one another in the
governing equations they solve, not just in implementation, so check which one
you are on before comparing results.

To add the upstream Alamo repository so you can pull in updates:

.. code-block::

    git remote add upstream https://github.com/solidsgroup/alamo.git
    git fetch upstream

Installing dependencies
-----------------------

FLAMES, like Alamo, is routinely run and tested on Ubuntu and MacOS. You can use
the Alamo
`System Install Scripts <https://solidsgroup.github.io/alamo/docs/GettingStarted.html#system-install-scripts>`_
to install all necessary dependencies for your system.

Setting default MPI
-------------------

It may be necessary to use a specific MPI distribution.
On Ubuntu, you can change the distribution with the following:

.. code-block::

    $> sudo update-alternatives --config mpi

    There are 2 choices for the alternative mpi (providing /usr/bin/mpicc).

      Selection    Path                    Priority   Status
    ------------------------------------------------------------
    * 0            /usr/bin/mpicc.openmpi   50        auto mode
      1            /usr/bin/mpicc.mpich     40        manual mode
      2            /usr/bin/mpicc.openmpi   50        manual mode

    Press <enter> to keep the current choice[*], or type selection number:

Do the same thing for mpirun.

.. code-block::

    $> sudo update-alternatives --config mpirun

Remember to run :code:`make realclean` every time you switch mpi versions.

Configuring
-----------

To compile FLAMES, you must first run the configure script.
This is done simply by running the following in the flames directory
(note that AMReX download is triggered by this command, so it may take a couple
minutes to complete depending on your internet connection)

.. code-block::

    ./configure

By default, this configures in 3D production mode.
For the multiphase solver, configure in 2D and pull in Eigen:

.. code-block::

    ./configure --dim 2 --get-eigen

To compile in 2D debug mode,

.. code-block::

    ./configure --dim=2 --debug

There are multiple compilation options available, and they must all be specified
at configure time. For a complete listing of the configuration options, type

.. code-block::

    ./configure --help

.. NOTE::
    The configure script produces output designed to assist in determining
    compile issues. Whenever you request help, please always include the
    complete output of the configure script.

Compiling
---------

Once you have configured, compile by

.. code-block::

    make

If you are on a platform with multiple cores, you can compile in parallel (for
instance, with 4 cores) with

.. code-block::

    make -j4

Executables are stored in :code:`./bin/`, named according to the driver and the
options specified at configure time. The multiphase driver is
:code:`src/hydro2.cc`, so a 2D debug build produces
:code:`./bin/hydro2-2d-debug-g++`; a 2D production build produces
:code:`./bin/hydro2-2d-g++`. The Alamo drivers (:code:`alamo`,
:code:`mechanics`, :code:`hydro`, :code:`sfi`, :code:`fracture`,
:code:`thermoelastic`, :code:`topop`) are built alongside it.

You can work with multiple versions at the same time without having to
re-compile the entire code base. All you need to do is re-run the configure
script, and previous versions of FLAMES and AMReX will be saved automatically.

.. WARNING::
    There is an issue with GNU Make that can cause I/O errors during parallel
    builds. You may get the following error:

    .. code-block::

        make[1]: write error: stdout

    To continue the build, just issue the :code:`make` command again and it
    should continue normally. You can also add the :code:`--output-sync=target`
    option which may help eliminate the issue.

Running
-------

Every case is driven by an input file. Point the executable at one:

.. code-block::

    ./bin/hydro2-2d-g++ tests/StefanVap/Saturation_Box_1D/input

or in parallel,

.. code-block::

    mpirun -np 8 ./bin/hydro2-2d-g++ tests/FlowShockDroplet/Air-Dodecane-Ma6_5/input

Any input parameter can be overridden on the command line by appending
:code:`name=value` pairs. SLURM launchers for the larger droplet cases are at
the repository root (:code:`run_droplet.slurm`, :code:`run_small_droplet.slurm`,
:code:`run_droplet_Angled.slurm`).

Unit Testing
------------

Upon successful compilation, run tests by

.. code-block::

    make test

This will run the unit tests and regression tests for all compiled production
versions. If you have only run in 2D, only 2D tests will be generated.
If you are a developer and you are preparing to merge your branch into
:code:`development`, you should perform a complete test via

.. code-block::

    ./configure --dim=2
    make
    ./configure --dim=3
    make
    make test

The multiphase verification cases under :code:`tests/StefanVap` ship with
Python checkers (:code:`check_stefan.py`, :code:`check_saturation_box.py`,
:code:`check_conduction_2phase.py`, :code:`check_latent.py`, and others) that
compare a completed run against the analytical or conservation-based
expectation.

Citing
------

FLAMES is built on Alamo. If you use FLAMES, please cite the Alamo paper:

.. code-block::

    @article{runnels2025alamo,
      title={The Alamo multiphysics solver for phase field simulations with strong-form mechanics and block structured adaptive mesh refinement},
      author={Runnels, Brandon and Agrawal, Vinamra and Meier, Maycon},
      journal={Journal of Open Source Software},
      year={2025}
    }

If you use the diffuse-boundary multiphase capability, please also cite the
method papers:

.. code-block::

    @article{schmidt2022selfsimilar,
      title={Self-similar Diffuse Boundary Method for Phase Boundary Driven Flow},
      author={Schmidt, E. M. and Quinlan, J. M. and Runnels, B.},
      journal={Physics of Fluids},
      volume={34},
      number={11},
      year={2022}
    }

    @article{boyd2026diffuse,
      title={A diffuse boundary method for phase boundaries in viscous compressible flow},
      author={Boyd, E. and Sandall, E. and Meier, M. and Quinlan, J. M. and Runnels, B.},
      journal={Journal of Computational Physics},
      volume={559},
      number={114898},
      year={2026}
    }

License
-------

MIT, inherited from Alamo. See :code:`LICENSE`.
