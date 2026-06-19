//
// Initialize a field (e.g. the embedded-solid indicator phi) from a 3D STL
// triangle mesh.  Wraps AMReX's STLtools, which reads the STL (ASCII or binary),
// builds a BVH, and returns the SIGNED DISTANCE to the surface:
//
//     d  >  0   OUTSIDE the mesh  (fluid)
//     d  <  0   INSIDE  the mesh  (solid)
//
// which is mapped to a diffuse indicator
//
//     phi = 0.5 * (1 + tanh(d / eps))        (1 = fluid, 0 = solid; band ~ eps)
//
// REQUIREMENTS: a 3D build (AMREX_SPACEDIM == 3) compiled with USE_EB=TRUE
// (STLtools lives in AMReX's embedded-boundary module).  This file is a no-op
// in any other build, so it is always safe to include.  Note that ONLY the
// STL reader + distance query are borrowed from EB -- the solver never builds
// an EB index space, so the eb2.* input language is never used; the mesh is
// configured purely through this IC's own keys (e.g. solid.phi.ic.stl.*).
//

#ifndef IC_STL_H
#define IC_STL_H

#include <AMReX_Config.H>   // AMREX_SPACEDIM, AMREX_USE_EB

#if defined(AMREX_USE_EB) && (AMREX_SPACEDIM == 3)

#include <cmath>
#include <array>
#include <AMReX_EB_STL_utils.H>

#include "IC/IC.H"
#include "IO/ParmParse.H"
#include "Set/Set.H"
#include "Util/Util.H"

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
        // (1) Fill the field MultiFab (valid + ghost cells) with the signed
        //     distance to the STL surface.  d > 0 fluid, d < 0 solid.
        stl.fillSignedDistance(*a_field[lev], a_field[lev]->nGrowVect(), geom[lev]);

        // (2) Map the signed distance to the diffuse indicator in place.
        const Set::Scalar eps_loc = eps;
        for (amrex::MFIter mfi(*a_field[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box bx = mfi.growntilebox();
            amrex::Array4<Set::Scalar> const &field = a_field[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                field(i, j, k, 0) = 0.5 * (1.0 + std::tanh(field(i, j, k, 0) / eps_loc));
                if (field.nComp() > 1) field(i, j, k, 1) = 1.0 - field(i, j, k, 0);
            });
        }
        a_field[lev]->FillBoundary();
    }

private:
    amrex::STLtools stl;
    Set::Scalar eps = 0.02;   // diffuse-interface half-width (in domain units)

public:
    static void Parse(STL &value, IO::ParmParse &pp)
    {
        std::string filename;
        pp_query_file("filename", filename);                    // STL file (ASCII or binary)

        Set::Scalar scale = 1.0;
        pp_query_default("scale", scale, 1.0);                  // uniform scale applied to the mesh

        Set::Vector center = Set::Vector::Zero();
        pp_queryarr("center", center);                         // mesh placement offset (see AMReX read_stl_file)

        int reverse_normal = 0;
        pp_query_default("reverse_normal", reverse_normal, 0); // flip inside/outside if the solid comes out inverted

        pp_query_default("eps", value.eps, 0.02);              // diffuse-interface half-width (~ a few dx_finest)

        // STLtools reads, scales, places, and builds its own BVH up front.
        value.stl.read_stl_file(filename, scale,
                                {center(0), center(1), center(2)}, reverse_normal);
    }
};
}

#endif // AMREX_USE_EB && AMREX_SPACEDIM == 3
#endif // IC_STL_H
