//
// This initializes and runs the two-phase Alamo hydrodynamic solver implemented in the
// :ref:`Integrator::Hydro2` integrator.
//

#include <iostream>
#include <fstream>
#include <iomanip>

#include "Util/Util.H"
#include "IO/ParmParse.H"
#include "IO/FileNameParse.H"
#include "IO/WriteMetaData.H"
#include "AMReX_ParmParse.H"

#include "Integrator/Hydro2.H"

int main (int argc, char* argv[])
{
    Util::Initialize(argc,argv);

    IO::ParmParse pp;
    srand(2);

    Integrator::Integrator *integrator = nullptr;
    pp.select_only<Integrator::Hydro2>(integrator);

    integrator->InitData();
    integrator->Evolve();
    delete integrator;

    Util::Finalize();
}
