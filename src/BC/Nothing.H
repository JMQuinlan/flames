//
// Empty BC class, used when specifying boundaries that have no
// ghost cells/nodes but a BC object is still required.
//
// NOT QUITE A NO-OP ANY MORE (2026-09-04): a BC object registered with a fab
// is also the physbc AMReX FillPatch / InterpFromCoarseLevel consults, and the
// conservative-interp slope stencils UNCONDITIONALLY read the out-of-domain
// neighbor of every domain-edge coarse cell (AMReX_MFInterp_C.H
// mf_compute_slopes_*; the centered slope reads it, and the ext_dir/hoextrap
// one-sided variants read it too).  A true no-op fill therefore feeds
// whatever the heap held into the interpolation the moment a fine level
// touches a physical boundary -- measured as snan traps under
// amrex.init_snan=1 and, unpoisoned, as garbage regridded cells at NSCBC
// faces (FlowMarmottant sweep, EOS-floor breach at t~4.7e-4).  FillBoundary
// now clamp-copies the nearest in-domain cell into out-of-domain ghosts
// (zero-gradient): still "nothing" as physics, but every cell the
// interpolator can read is finite.  GetBCRec reports foextrap to match.
//

#ifndef BC_NOTHING_H_
#define BC_NOTHING_H_

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_BCRec.H>
#include <AMReX_PhysBCFunct.H>

#include "BC/BC.H"

namespace BC
{
class Nothing
    : public BC<Set::Scalar>
{

public:
    Nothing () {};
    virtual void FillBoundary (amrex::BaseFab<Set::Scalar>& a_in,
                    const amrex::Box &a_box,
                    int ngrow, int /*dcomp*/, int /*ncomp*/,
                    amrex::Real /*time*/,
                    Orientation /*face*/ = Orientation::All,
                    const amrex::Mask * /*mask*/ = nullptr) override
    {
        // Cell-centered fabs only: a node fab's boundary-face nodes share the
        // out-of-cell-domain index and the clamp would overwrite valid nodes.
        // Node fields registered with BC::Nothing keep the historical no-op.
        if (!a_in.box().ixType().cellCentered()) return;
        amrex::Box box = a_box;
        box.grow(ngrow);
        const amrex::Dim3 lo = amrex::lbound(m_geom.Domain());
        const amrex::Dim3 hi = amrex::ubound(m_geom.Domain());
        amrex::Array4<amrex::Real> const& in = a_in.array();
        const int ncomp_fab = a_in.nComp();
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            int ic = amrex::min(amrex::max(i, lo.x), hi.x);
            int jc = amrex::min(amrex::max(j, lo.y), hi.y);
            int kc = k;
#if AMREX_SPACEDIM == 3
            kc = amrex::min(amrex::max(k, lo.z), hi.z);
#endif
            if (ic == i && jc == j && kc == k) return;   // in-domain: untouched
            for (int n = 0; n < ncomp_fab; ++n)
                in(i, j, k, n) = in(ic, jc, kc, n);
        });
    };

    virtual amrex::BCRec GetBCRec() override
    {
        // foextrap, NOT default-constructed (= BCType::bogus): see header note.
        return amrex::BCRec(AMREX_D_DECL(amrex::BCType::foextrap, amrex::BCType::foextrap, amrex::BCType::foextrap),
                            AMREX_D_DECL(amrex::BCType::foextrap, amrex::BCType::foextrap, amrex::BCType::foextrap));
    }
};
}
#endif
