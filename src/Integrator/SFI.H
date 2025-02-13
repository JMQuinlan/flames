#if AMREX_SPACEDIM==2

#ifndef INTEGRATOR_SFI_H
#define INTEGRATOR_SFI_H

#include "Integrator/AllenCahn.H"
#include "Integrator/Hydro.H"

namespace Integrator
{
class SFI : 
    virtual public AllenCahn,
    virtual public Hydro
{
public:
    SFI():
        AllenCahn(),
        Hydro()
    {
    }
    SFI(IO::ParmParse &pp) : SFI()
    {Parse(*this,pp);}
    static void Parse(SFI &value, IO::ParmParse &pp)
    {
        pp.queryclass("allencahn",static_cast<AllenCahn*>(&value));
        pp.queryclass("hydro",static_cast<Hydro*>(&value));

    }


protected:
    void Initialize(int lev) override
    {
        AllenCahn::Initialize(lev);
        AllenCahn::ic->Initialize(lev,eta_mf,0.0);
        AllenCahn::ic->Initialize(lev,eta_old_mf,0.0);
        Hydro::Initialize(lev);
    }

    void TimeStepBegin(Set::Scalar a_time, int a_step) override
    {
        if (a_time > tstart && !hydro_initialized)
        {
            for (int lev = 0; lev <= finest_level; lev++)
            {
                Hydro::Initialize(lev);
                Hydro::Mix(lev);
            }
            hydro_initialized=true;
        }
        AllenCahn::TimeStepBegin(a_time, a_step);
        if (hydro_initialized) Hydro::TimeStepBegin(a_time, a_step);
    }

    void Advance(int a_lev, amrex::Real a_time, amrex::Real a_dt) override
    {
        AllenCahn::Advance(a_lev, a_time, a_dt);
        if (hydro_initialized) Hydro::Advance(a_lev, a_time, a_dt);
    }

    void TagCellsForRefinement(int a_lev, amrex::TagBoxArray& a_tags, Set::Scalar a_time, int a_ngrow) override
    {
        AllenCahn::TagCellsForRefinement(a_lev, a_tags, a_time, a_ngrow);
        if (hydro_initialized) Hydro::TagCellsForRefinement(a_lev, a_tags, a_time, a_ngrow);
    }

    void UpdateEta(int lev, Set::Scalar /*time*/) override
    {
        for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.growntilebox();
            amrex::Array4<Set::Scalar> const& eta_old = (*eta_old_mf[lev]).array(mfi);
            amrex::Array4<Set::Scalar> const& eta_new = (*eta_mf[lev]).array(mfi);
            amrex::Array4<const Set::Scalar> const& alpha_old = (*alpha_old_mf[lev]).array(mfi);
            amrex::Array4<const Set::Scalar> const& alpha_new = (*alpha_mf[lev]).array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                eta_new(i,j,k) = alpha_new(i,j,k);
                eta_old(i,j,k) = alpha_old(i,j,k);
            });
        }
        Util::RealFillBoundary(*eta_mf[lev], geom[lev]);
        Util::RealFillBoundary(*eta_old_mf[lev], geom[lev]);
    }

    int invert = false;
    Set::Scalar tstart = 1.0;
    bool hydro_initialized = false;
};
} // namespace Integrator
#endif

#endif
