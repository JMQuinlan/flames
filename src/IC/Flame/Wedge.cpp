#include <AMReX_ParallelDescriptor.H>
#include "Wedge.H"

namespace IC
{
namespace Flame
{
Wedge::Wedge (amrex::Vector<amrex::Geometry> &_geom)
	: IC(_geom)
{
}
  
void Wedge::Initialize(const int lev, amrex::Vector<std::unique_ptr<amrex::MultiFab> > &field)
{
	AMREX_D_TERM(amrex::Real sizex = geom[0].ProbHi()[0] - geom[0].ProbLo()[0];,
		     amrex::Real sizey = geom[0].ProbHi()[1] - geom[0].ProbLo()[1];,
		     amrex::Real sizez = geom[0].ProbHi()[2] - geom[0].ProbLo()[2];);
	

	amrex::Real angle = 45.0 * 0.01745329251;


	for (amrex::MFIter mfi(*field[lev],true); mfi.isValid(); ++mfi)
	{
		const amrex::Box& box = mfi.tilebox();

		amrex::BaseFab<amrex::Real> &field_box = (*field[lev])[mfi];

		AMREX_D_TERM(for (int i = box.loVect()[0]-field[lev]->nGrow(); i<=box.hiVect()[0]+field[lev]->nGrow(); i++),
			     for (int j = box.loVect()[1]-field[lev]->nGrow(); j<=box.hiVect()[1]+field[lev]->nGrow(); j++),
			     for (int k = box.loVect()[2]-field[lev]->nGrow(); k<=box.hiVect()[2]+field[lev]->nGrow(); k++))
		{
			AMREX_D_TERM(amrex::Real x = geom[lev].ProbLo()[0] + ((amrex::Real)(i) + 0.5) * geom[lev].CellSize()[0];,
				     amrex::Real y = geom[lev].ProbLo()[1] + ((amrex::Real)(j) + 0.5) * geom[lev].CellSize()[1];,
				     amrex::Real z = geom[lev].ProbLo()[2] + ((amrex::Real)(k) + 0.5) * geom[lev].CellSize()[2];);

			amrex::IntVect m(AMREX_D_DECL(i,j,k));


			if (y > (x+0.5*sizex)*std::tan(angle) ||
			    y < -(x+0.5*sizex)*std::tan(angle)) // outside
				field_box(m) = 0.0;
			else
				field_box(m) = 1.0;

		}
	}
}
}
}
