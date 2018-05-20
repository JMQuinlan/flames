#include <AMReX_ParallelDescriptor.H>
#include <Constant.H>

namespace IC
{

Constant::Constant(amrex::Vector<amrex::Geometry> &_geom, amrex::Vector<amrex::Real> _value)
	: IC(_geom), value(_value) {}

void Initialize(const int lev,
		amrex::Vector<std::unique_ptr<amrex::MultiFab> > &field)
{
	for (amrex::MFIter mfi(*field[lev],true); mfi.isValid(); ++mfi)
	{
		const amrex::Box& box = mfi.tilebox();

		amrex::FArrayBox &field_box = (*field[lev])[mfi];

		for (int i = box.loVect()[0]-field[lev]->nGrow(); i<=box.hiVect()[0]+field[lev]->nGrow(); i++) 
			for (int j = box.loVect()[1]-field[lev]->nGrow(); j<=box.hiVect()[1]+field[lev]->nGrow(); j++)
#if BL_SPACEDIM==3
				for (int k = box.loVect()[2]-field[lev]->nGrow(); k<=box.hiVect()[2]+field[lev]->nGrow(); k++)
#endif
			{
				for (int m = 0; m < value.size(); m++)
					field_box(amrex::IntVect(AMREX_D_DECL(i,j,k)),m) = value[m];     
			}
	}

}
	


}
