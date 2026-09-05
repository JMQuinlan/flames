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

    // ---------------------------------------------------------------------
    // PROPER-NESTING AUTO-DEFAULT (must run BEFORE the integrator is
    // constructed -- amrex::AmrMesh reads amr.n_proper in its constructor).
    //
    // FillPatch of a fine level's ghost region interpolates from the PARENT
    // level and needs the parent's valid data to cover the fine box grown by
    // nghost (coarsened by the ref ratio) plus one coarse cell for the
    // conservative-interp slope stencil: required = nghost/ref + 1 coarse
    // cells.  AMReX guarantees a nesting width of only n_proper *
    // blocking_factor / ref coarse cells, so small blocking factors with the
    // default n_proper = 1 let ghost regions poke past the parent -- and
    // FillPatchTwoLevels then SILENTLY interpolates from UNFILLED memory
    // (measured: snan traps in amrex::mf_cell_cons_lin_interp at
    // blocking_factor 2 and 4 with nghost = 4; heap garbage in production).
    // Rather than demanding input edits, default n_proper to the value that
    // restores the guarantee:  n_proper = ceil(required * ref / bf).  For
    // blocking_factor >= 2*nghost this evaluates to 1 -- existing safe decks
    // regrid EXACTLY as before.  An explicit amr.n_proper in the input wins.
    {
        amrex::ParmParse pp_amr("amr");
        if (!pp_amr.contains("n_proper"))
        {
            int nghost = 2;
            pp.query("nghost", nghost);
            std::vector<int> bfv;
            pp_amr.queryarr("blocking_factor", bfv);
            int bf = 8;                       // AMReX default when unset
            for (int b : bfv) bf = std::min(bf, b);
            const int ref = 2;                // ref_ratio 2 throughout hydro2
            const int required = nghost / ref + 1;              // coarse cells
            const int nproper = std::max(1, (required * ref + bf - 1) / bf);
            pp_amr.add("n_proper", nproper);
            if (nproper > 1)
                Util::Message(INFO, "amr.n_proper auto-set to ", nproper,
                              " (blocking_factor=", bf, ", nghost=", nghost,
                              "): guarantees parent coverage for coarse-fine ghost fills");
        }
    }

    Integrator::Integrator *integrator = nullptr;
    pp.select_only<Integrator::Hydro2>(integrator);

    integrator->InitData();
    integrator->Evolve();
    delete integrator;

    Util::Finalize();
}
