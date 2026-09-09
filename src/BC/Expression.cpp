#include "Expression.H"

namespace BC
{

void
Expression::FillBoundary (amrex::BaseFab<Set::Scalar> &a_in,
                        const amrex::Box &a_box,
                        int ngrow, int /*dcomp*/, int /*ncomp*/, Set::Scalar time,
                        Orientation face, const amrex::Mask * /*mask*/)
{
    const amrex::Real* DX = m_geom.CellSize();

    Util::Assert(INFO,TEST(a_in.nComp() == (int)m_ncomp));

    amrex::Box box = a_box;
    box.grow(ngrow);
    const amrex::Dim3 lo= amrex::lbound(m_geom.Domain()), hi = amrex::ubound(m_geom.Domain());

    amrex::Array4<amrex::Real> const& in = a_in.array();

    amrex::IndexType type = amrex::IndexType::TheCellType();

    for (int n = 0; n < a_in.nComp(); n++)
    amrex::ParallelFor (box,[=] AMREX_GPU_DEVICE(int i, int j, int k)
    {
        Set::Vector pos = Set::Position(i, j, k, m_geom, type);
        Set::Scalar x = 0.0, y=0.0, z=0.0, t=time;
        x = pos(0);
        #if AMREX_SPACEDIM > 1
        y = pos(1);
        #if AMREX_SPACEDIM > 2
        z = pos(2);
        #endif
        #endif


        amrex::IntVect glevel;
        AMREX_D_TERM(glevel[0] = std::max(std::min(0,i-lo.x),i-hi.x); ,
                    glevel[1] = std::max(std::min(0,j-lo.y),j-hi.y); ,
                    glevel[2] = std::max(std::min(0,k-lo.z),k-hi.z); );
        
        // --------------------------------------------------------------
        // COMPOSED EDGE/CORNER GHOSTS (2026-09-09).  A ghost outside TWO or
        // more domain faces only entered the FIRST matching single-face
        // branch below, mirroring/copying from a cell that is itself still a
        // ghost on the other axis -- an order-dependent, one-fill-stale read
        // (same hazard class as the NSCBC4 corner bug; measured as part of
        // the symmetry-AXIS over-response in the driven-bubble runs).  When
        // every out-of-domain axis is a mirror- or zero-gradient-type face,
        // resolve them all at once: per-axis mirrored/clamped source index,
        // sign = product of REFLECT_ODD crossings.  Exact for the mirror
        // composition, order-independent, reads interior data only.
        // Non-composable types (dirichlet et al.) fall through to the
        // legacy single-face branches.
        {
            int nout = 0;
            bool composable = true;
            Set::Scalar csign = 1.0;
            int si = i, sj = j, sk = k;
            if (glevel[0] != 0)
            {
                nout++;
                const int tt = (glevel[0] < 0) ? m_bc_type[Face::XLO][n] : m_bc_type[Face::XHI][n];
                if (BCUtil::IsReflectEven(tt))      si = (glevel[0] < 0) ? 2*lo.x - 1 - i : 2*hi.x + 1 - i;
                else if (BCUtil::IsReflectOdd(tt)) { si = (glevel[0] < 0) ? 2*lo.x - 1 - i : 2*hi.x + 1 - i; csign = -csign; }
                else if (BCUtil::IsNeumann(tt) || BCUtil::IsNSCBC(tt)) si = i - glevel[0];
                else composable = false;
            }
#if AMREX_SPACEDIM > 1
            if (glevel[1] != 0)
            {
                nout++;
                const int tt = (glevel[1] < 0) ? m_bc_type[Face::YLO][n] : m_bc_type[Face::YHI][n];
                if (BCUtil::IsReflectEven(tt))      sj = (glevel[1] < 0) ? 2*lo.y - 1 - j : 2*hi.y + 1 - j;
                else if (BCUtil::IsReflectOdd(tt)) { sj = (glevel[1] < 0) ? 2*lo.y - 1 - j : 2*hi.y + 1 - j; csign = -csign; }
                else if (BCUtil::IsNeumann(tt) || BCUtil::IsNSCBC(tt)) sj = j - glevel[1];
                else composable = false;
            }
#endif
#if AMREX_SPACEDIM > 2
            if (glevel[2] != 0)
            {
                nout++;
                const int tt = (glevel[2] < 0) ? m_bc_type[Face::ZLO][n] : m_bc_type[Face::ZHI][n];
                if (BCUtil::IsReflectEven(tt))      sk = (glevel[2] < 0) ? 2*lo.z - 1 - k : 2*hi.z + 1 - k;
                else if (BCUtil::IsReflectOdd(tt)) { sk = (glevel[2] < 0) ? 2*lo.z - 1 - k : 2*hi.z + 1 - k; csign = -csign; }
                else if (BCUtil::IsNeumann(tt) || BCUtil::IsNSCBC(tt)) sk = k - glevel[2];
                else composable = false;
            }
#endif
            if (nout >= 2 && composable)
            {
                in(i, j, k, n) = csign * in(si, sj, sk, n);
                return;
            }
        }

        if (glevel[0]<0 && (face == Orientation::xlo || face == Orientation::All)) // Left boundary
        {
            if (BCUtil::IsDirichlet(m_bc_type[Face::XLO][n]))
                in(i,j,k,n) = m_bc_func[Face::XLO][n](x,y,z,t);
            else if(BCUtil::IsNeumann(m_bc_type[Face::XLO][n]))
                in(i,j,k,n) = in(i-glevel[0],j,k,n) - (m_bc_func[Face::XLO].size() > 0 ? m_bc_func[Face::XLO][n](x,y,z,t)*DX[0] : 0);
            // Mirror across the lo face (at lo.x-1/2): ghost i -> interior 2*lo.x-1-i.
            else if(BCUtil::IsReflectEven(m_bc_type[Face::XLO][n]))
                in(i,j,k,n) = in(2*lo.x - 1 - i,j,k,n);
            else if(BCUtil::IsReflectOdd(m_bc_type[Face::XLO][n]))
                in(i,j,k,n) = -in(2*lo.x - 1 - i,j,k,n);
            else if(BCUtil::IsPeriodic(m_bc_type[Face::XLO][n])) {}
            // NSCBC-typed faces: clamp-copy (zero-gradient) so this BC is a
            // SAFE physbc for FillPatch/interp (finite out-of-domain data);
            // the real characteristic ghosts are written by NSCBC4 afterwards.
            else if (BCUtil::IsNSCBC(m_bc_type[Face::XLO][n]))
                in(i,j,k,n) = in(i-glevel[0],j,k,n);
            else
                Util::Abort(INFO, "Incorrect boundary conditions");
        }
        else if (glevel[0]>0 && (face == Orientation::xhi || face == Orientation::All)) // Right boundary
        {
            if (BCUtil::IsDirichlet(m_bc_type[Face::XHI][n]))
                in(i,j,k,n) = m_bc_func[Face::XHI][n](x,y,z,t);
            else if(BCUtil::IsNeumann(m_bc_type[Face::XHI][n]))
                in(i,j,k,n) = in(i-glevel[0],j,k,n) - (m_bc_func[Face::XHI].size() > 0 ? m_bc_func[Face::XHI][n](x,y,z,t)*DX[0] : 0);
            // Mirror across the hi face (at hi.x+1/2): ghost i -> interior 2*hi.x+1-i.
            else if(BCUtil::IsReflectEven(m_bc_type[Face::XHI][n]))
                in(i,j,k,n) = in(2*hi.x + 1 - i,j,k,n);
            else if(BCUtil::IsReflectOdd(m_bc_type[Face::XHI][n]))
                in(i,j,k,n) = -in(2*hi.x + 1 - i,j,k,n);
            else if(BCUtil::IsPeriodic(m_bc_type[Face::XHI][n])) {}
            // NSCBC-typed faces: clamp-copy (zero-gradient) so this BC is a
            // SAFE physbc for FillPatch/interp (finite out-of-domain data);
            // the real characteristic ghosts are written by NSCBC4 afterwards.
            else if (BCUtil::IsNSCBC(m_bc_type[Face::XHI][n]))
                in(i,j,k,n) = in(i-glevel[0],j,k,n);
            else
                Util::Abort(INFO, "Incorrect boundary conditions");
        }
        
        else if (glevel[1]<0 && (face == Orientation::ylo || face == Orientation::All)) // Bottom boundary
        {
            if (BCUtil::IsDirichlet(m_bc_type[Face::YLO][n]))
                in(i,j,k,n) = m_bc_func[Face::YLO][n](x,y,z,t);
            else if (BCUtil::IsNeumann(m_bc_type[Face::YLO][n]))
                in(i,j,k,n) = in(i,j-glevel[1],k,n) - (m_bc_func[Face::YLO].size() > 0 ? m_bc_func[Face::YLO][n](x,y,z,t)*DX[1] : 0);
            else if (BCUtil::IsReflectEven(m_bc_type[Face::YLO][n]))
                in(i,j,k,n) = in(i,2*lo.y - 1 - j,k,n);
            else if (BCUtil::IsReflectOdd(m_bc_type[Face::YLO][n]))
                in(i,j,k,n) = -in(i,2*lo.y - 1 - j,k,n);
            else if(BCUtil::IsPeriodic(m_bc_type[Face::YLO][n])) {}
            // NSCBC-typed faces: clamp-copy (zero-gradient) so this BC is a
            // SAFE physbc for FillPatch/interp (finite out-of-domain data);
            // the real characteristic ghosts are written by NSCBC4 afterwards.
            else if (BCUtil::IsNSCBC(m_bc_type[Face::YLO][n]))
                in(i,j,k,n) = in(i,j-glevel[1],k,n);
            else
                Util::Abort(INFO, "Incorrect boundary conditions");
        }
        else if (glevel[1]>0 && (face == Orientation::yhi || face == Orientation::All)) // Top boundary
        {
            if (BCUtil::IsDirichlet(m_bc_type[Face::YHI][n]))
                in(i,j,k,n) = m_bc_func[Face::YHI][n](x,y,z,t);
            else if (BCUtil::IsNeumann(m_bc_type[Face::YHI][n]))
                in(i,j,k,n) = in(i,j-glevel[1],k,n) - (m_bc_func[Face::YHI].size() > 0 ? m_bc_func[Face::YHI][n](x,y,z,t)*DX[1] : 0);
            else if (BCUtil::IsReflectEven(m_bc_type[Face::YHI][n]))
                in(i,j,k,n) = in(i,2*hi.y + 1 - j,k,n);
            else if (BCUtil::IsReflectOdd(m_bc_type[Face::YHI][n]))
                in(i,j,k,n) = -in(i,2*hi.y + 1 - j,k,n);
            else if(BCUtil::IsPeriodic(m_bc_type[Face::YHI][n])) {}
            // NSCBC-typed faces: clamp-copy (zero-gradient) so this BC is a
            // SAFE physbc for FillPatch/interp (finite out-of-domain data);
            // the real characteristic ghosts are written by NSCBC4 afterwards.
            else if (BCUtil::IsNSCBC(m_bc_type[Face::YHI][n]))
                in(i,j,k,n) = in(i,j-glevel[1],k,n);
            else
                Util::Abort(INFO, "Incorrect boundary conditions");
        }

#if AMREX_SPACEDIM>2
        else if (glevel[2]<0 && (face == Orientation::zlo || face == Orientation::All))
        {
            if (BCUtil::IsDirichlet(m_bc_type[Face::ZLO][n]))
                in(i,j,k,n) = m_bc_func[Face::ZLO][n](x,y,z,t);
            else if (BCUtil::IsNeumann(m_bc_type[Face::ZLO][n]))
                in(i,j,k,n) = in(i,j,k-glevel[2],n) - (m_bc_func[Face::ZLO].size() > 0 ? m_bc_func[Face::ZLO][n](x,y,z,t)*DX[2] : 0);
            else if (BCUtil::IsReflectEven(m_bc_type[Face::ZLO][n]))
                in(i,j,k,n) = in(i,j,2*lo.z - 1 - k,n);
            else if (BCUtil::IsReflectOdd(m_bc_type[Face::ZLO][n]))
                in(i,j,k,n) = -in(i,j,2*lo.z - 1 - k,n);
            else if(BCUtil::IsPeriodic(m_bc_type[Face::ZLO][n])) {}
            // NSCBC-typed faces: clamp-copy (zero-gradient) so this BC is a
            // SAFE physbc for FillPatch/interp (finite out-of-domain data);
            // the real characteristic ghosts are written by NSCBC4 afterwards.
            else if (BCUtil::IsNSCBC(m_bc_type[Face::ZLO][n]))
                in(i,j,k,n) = in(i,j,k-glevel[2],n);
            else Util::Abort(INFO, "Incorrect boundary conditions");
        }
        else if (glevel[2]>0 && (face == Orientation::zhi || face == Orientation::All))
        {
            if (BCUtil::IsDirichlet(m_bc_type[Face::ZHI][n]))
                in(i,j,k,n) = m_bc_func[Face::ZHI][n](x,y,z,t);
            else if(BCUtil::IsNeumann(m_bc_type[Face::ZHI][n]))
                in(i,j,k,n) = in(i,j,k-glevel[2],n) - (m_bc_func[Face::ZHI].size() > 0 ? m_bc_func[Face::ZHI][n](x,y,z,t)*DX[2] : 0);
            else if(BCUtil::IsReflectEven(m_bc_type[Face::ZHI][n]))
                in(i,j,k,n) = in(i,j,2*hi.z + 1 - k,n);
            else if(BCUtil::IsReflectOdd(m_bc_type[Face::ZHI][n]))
                in(i,j,k,n) = -in(i,j,2*hi.z + 1 - k,n);
            else if(BCUtil::IsPeriodic(m_bc_type[Face::ZHI][n])) {}
            // NSCBC-typed faces: clamp-copy (zero-gradient) so this BC is a
            // SAFE physbc for FillPatch/interp (finite out-of-domain data);
            // the real characteristic ghosts are written by NSCBC4 afterwards.
            else if (BCUtil::IsNSCBC(m_bc_type[Face::ZHI][n]))
                in(i,j,k,n) = in(i,j,k-glevel[2],n);
            else Util::Abort(INFO, "Incorrect boundary conditions");
        }
#endif


    });
}

amrex::BCRec
Expression::GetBCRec() 
{
    // Translate stored per-face types into ints amrex's FillPatch/interp
    // machinery understands.  NSCBC custom types (2000+) map to foextrap:
    // the ghost fill above clamp-copies them, so foextrap is the honest
    // description for slope/stencil purposes.  (Also fixes the z slots,
    // which previously reused the XLO/XHI types.)
    auto xlate = [](int t) -> int {
        if (BCUtil::IsNSCBC(t)) return (int)amrex::BCType::foextrap;
        return t;
    };
    int bc_lo[BL_SPACEDIM] = {AMREX_D_DECL(xlate(m_bc_type[Face::XLO][0]),
                                           xlate(m_bc_type[Face::YLO][0]),
                                           xlate(m_bc_type[Face::ZLO][0]))};
    int bc_hi[BL_SPACEDIM] = {AMREX_D_DECL(xlate(m_bc_type[Face::XHI][0]),
                                           xlate(m_bc_type[Face::YHI][0]),
                                           xlate(m_bc_type[Face::ZHI][0]))};

    return amrex::BCRec(bc_lo,bc_hi);
}

amrex::Array<int,AMREX_SPACEDIM>
Expression::IsPeriodic()
{
    return {AMREX_D_DECL(BCUtil::IsPeriodic(m_bc_type[Face::XLO][0]),
                BCUtil::IsPeriodic(m_bc_type[Face::YLO][0]),
                BCUtil::IsPeriodic(m_bc_type[Face::ZLO][0]))};
}
amrex::Periodicity Expression::Periodicity () const
{
    return amrex::Periodicity(amrex::IntVect(AMREX_D_DECL(m_geom.Domain().length(0) * BCUtil::IsPeriodic(m_bc_type[Face::XLO][0]),
                                                            m_geom.Domain().length(1) * BCUtil::IsPeriodic(m_bc_type[Face::YLO][0]),
                                                            m_geom.Domain().length(2) * BCUtil::IsPeriodic(m_bc_type[Face::ZLO][0]))));
}
amrex::Periodicity Expression::Periodicity (const amrex::Box& b) {
    return amrex::Periodicity(amrex::IntVect(AMREX_D_DECL(b.length(0) * BCUtil::IsPeriodic(m_bc_type[Face::XLO][0]),
                                                        b.length(1) * BCUtil::IsPeriodic(m_bc_type[Face::YLO][0]),
                                                        b.length(2) * BCUtil::IsPeriodic(m_bc_type[Face::ZLO][0]))));

}


}
