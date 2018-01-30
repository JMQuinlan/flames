
#include <AMReX_PlotFileUtil.H>

#include <PFFlame.H>

#include <fstream>


using namespace amrex;

std::string
PFFlame::PlotFileName (int lev) const
{
    return amrex::Concatenate(plot_file+"/", lev, 5);
}

Array<const MultiFab*>
PFFlame::PlotFileMF (int fab) const
{
    Array<const MultiFab*> r;
    for (int i = 0; i <= finest_level; ++i) {
	r.push_back(phi_new[fab][i].get());
    }
    return r;
}

Array<std::string>
PFFlame::PlotFileVarNames () const
{
  Array<std::string> names;
  // for (int n = 0; n < number_of_components; n++)
  //   names.push_back(amrex::Concatenate("phi",n));
  names.push_back("eta");
  names.push_back("T");
  return names;
}

void
PFFlame::WritePlotFile () const
{
    const std::string& plotfilename = PlotFileName(istep[0]);
    const auto& mf = PlotFileMF(0);
    const auto& varnames = PlotFileVarNames();
    amrex::WriteMultiLevelPlotfile(plotfilename, finest_level+1, mf, varnames,
				   Geom(), t_new[0], istep, refRatio());
    std::ofstream outfile;
    outfile.open(plot_file+".visit",std::ios_base::app);
    outfile << plotfilename + "/Header" << std::endl;
}

void
PFFlame::InitFromCheckpoint ()
{
    amrex::Abort("PFFlame::InitFromCheckpoint: todo");
}

