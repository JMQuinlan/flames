//
// Initialize a field (e.g. the embedded-solid indicator phi) from a 3D STL
// triangle mesh, using the SELF-CONTAINED Util::STL (Route B) -- no AMReX
// embedded-boundary module (USE_EB) required.  Util::STL reads the STL (binary
// or ASCII), builds a BVH, and returns the signed distance:
//
//     d > 0   OUTSIDE the mesh  (fluid)
//     d < 0   INSIDE  the mesh  (solid)
//
// mapped to a diffuse indicator
//
//     phi = 0.5 * (1 + tanh(d / eps))        (1 = fluid, 0 = solid; band ~ eps)
//
// 3D only (guarded on AMREX_SPACEDIM == 3; a no-op in 2D builds).  The mesh is
// configured purely through this IC's own keys (solid.phi.ic.stl.*).
//

#ifndef IC_STL_H
#define IC_STL_H

#include <AMReX_Config.H>   // AMREX_SPACEDIM

#if (AMREX_SPACEDIM == 3)

#include <cmath>
#include <array>

#include "IC/IC.H"
#include "IO/ParmParse.H"
#include "Set/Set.H"
#include "Util/Util.H"
#include "Util/STL.H"

namespace IC
{
class STL : public IC<Set::Scalar>
{
public:
    static constexpr const char* name = "stl";

    STL (amrex::Vector<amrex::Geometry> &_geom) : IC(_geom) {}

    STL (amrex::Vector<amrex::Geometry> &_geom, IO::ParmParse &pp, std::string a_name) : STL(_geom)
    {
        pp_queryclass(a_name, *this);
    }

    void Add(const int &lev, Set::Field<Set::Scalar> &a_field, Set::Scalar) override
    {
        const Util::STL *stlp = &stl;            // captured by value (pointer) into the lambda
        const Set::Scalar eps_loc = eps;
        const Set::Scalar *DX = geom[lev].CellSize();
        Set::Vector domlo(geom[lev].ProbLo()[0], geom[lev].ProbLo()[1], geom[lev].ProbLo()[2]);

        for (amrex::MFIter mfi(*a_field[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box bx = mfi.growntilebox();
            amrex::Array4<Set::Scalar> const &field = a_field[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                Set::Scalar x = domlo(0) + (i + 0.5) * DX[0];
                Set::Scalar y = domlo(1) + (j + 0.5) * DX[1];
                Set::Scalar z = domlo(2) + (k + 0.5) * DX[2];
                Set::Scalar d = stlp->SignedDistance(x, y, z);   // <0 solid, >0 fluid
                field(i, j, k, 0) = 0.5 * (1.0 + std::tanh(d / eps_loc));
                if (field.nComp() > 1) field(i, j, k, 1) = 1.0 - field(i, j, k, 0);
            });
        }
        a_field[lev]->FillBoundary();
    }

private:
    Util::STL  stl;
    Set::Scalar eps = 0.02;   // diffuse-interface half-width (domain units)

public:
    static void Parse(STL &value, IO::ParmParse &pp)
    {
        std::string filename;
        pp_query_file("filename", filename);                    // STL file (ASCII or binary)

        Set::Scalar scale = 1.0;
        pp_query_default("scale", scale, 1.0);                  // uniform scale applied to the mesh

        Set::Vector center = Set::Vector::Zero();
        pp_queryarr("center", center);                         // placement offset: vertex -> vertex*scale + center

        int reverse_normal = 0;
        pp_query_default("reverse_normal", reverse_normal, 0); // flip inside/outside if the solid comes out inverted

        pp_query_default("eps", value.eps, 0.02);              // diffuse-interface half-width (~ a few dx_finest)

        value.stl.Read(filename, scale, {center(0), center(1), center(2)}, reverse_normal);
    }
};
}

#endif // AMREX_SPACEDIM == 3
#endif // IC_STL_H
