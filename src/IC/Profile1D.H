//
// Initial condition from a TABULATED ONE-DIMENSIONAL PROFILE in a text file (linear interpolation
// along x, constant beyond the ends of the table).  Written for mapping a steady detonation (ZND)
// profile onto the grid, the usual way to start an unsteady detonation run.
//
// The file is plain text: lines starting with # are skipped, every other line is
// :code:`s  c1  c2  ...` with s increasing.  The table coordinate of a cell is
// :code:`s = x - x0`, or :code:`s = x0 - x` with :code:`flip = 1` (a profile given as the
// distance behind a front located at x0).  One column index (counted from 1 for c1) is given per
// component of the field; a negative index leaves that component at zero.
//

#ifndef IC_PROFILE1D_H_
#define IC_PROFILE1D_H_

#include <fstream>
#include <sstream>
#include "AMReX_Config.H"
#include "IC/IC.H"
#include "IO/ParmParse.H"
#include "Util/Util.H"

namespace IC
{
class Profile1D : public IC<Set::Scalar>
{
public:
    static constexpr const char* name = "profile1d";

    virtual ~Profile1D() = default;

    Profile1D (amrex::Vector<amrex::Geometry> &_geom) : IC<Set::Scalar>(_geom) {}
    Profile1D (amrex::Vector<amrex::Geometry> &_geom, IO::ParmParse &pp) : IC<Set::Scalar>(_geom)
    {pp_queryclass(*this);}
    Profile1D (amrex::Vector<amrex::Geometry> &_geom, IO::ParmParse &pp, std::string name) : IC<Set::Scalar>(_geom)
    {pp_queryclass(name,*this);}

    virtual void Add(const int &lev, Set::Field<Set::Scalar> &a_field, Set::Scalar /*time*/) override
    {
        const int nc = a_field[lev]->nComp();
        Util::Assert(INFO,TEST((int)m_columns.size() >= 1 && (int)m_columns.size() <= nc));
        const Set::Scalar xlo = geom[lev].ProbLo()[0], dx = geom[lev].CellSize()[0];
        const int ns = (int)m_s.size(), ncol = m_ncol, nm = (int)m_columns.size();
        const Set::Scalar *s = m_s.data(), *v = m_v.data();
        const int *col = m_columns.data();
        const Set::Scalar x0 = m_x0; const int flip = m_flip;
        for (amrex::MFIter mfi(*a_field[lev],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.growntilebox();
            amrex::Array4<Set::Scalar> const& field = a_field[lev]->array(mfi);
            amrex::ParallelFor (bx,[=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar x = xlo + (i + 0.5) * dx, sc = flip ? x0 - x : x - x0;
                int a = 0, b = ns - 1;                          // bracket sc in the table
                Set::Scalar w = 0.0;
                if (sc <= s[0]) { a = b = 0; }
                else if (sc >= s[ns - 1]) { a = b = ns - 1; }
                else
                {
                    while (b - a > 1) { const int m = (a + b) / 2; if (s[m] <= sc) a = m; else b = m; }
                    w = (sc - s[a]) / (s[b] - s[a]);
                }
                for (int m = 0; m < nm; ++m)
                    if (col[m] >= 1)
                        field(i,j,k,m) += (1.0 - w) * v[a * ncol + col[m] - 1] + w * v[b * ncol + col[m] - 1];
            });
        }
    }

    static void Parse(Profile1D & value, IO::ParmParse & pp)
    {
        std::string file;
        // Text file with the profile: s c1 c2 ... per line
        pp_query_required("file", file);
        // Column (1 = first after s) for each component of the field; negative = leave at zero
        pp.queryarr_required("columns", value.m_columns);
        // Position of the table origin s = 0 in the domain
        pp_query_default("x0", value.m_x0, 0.0);
        // 1: the table coordinate is the distance BEHIND x0 (s = x0 - x)
        pp_query_default("flip", value.m_flip, 0);

        std::ifstream in(file);
        if (!in.good()) Util::Abort(INFO, "IC::Profile1D: cannot open ", file);
        std::string line;
        value.m_ncol = -1;
        while (std::getline(in, line))
        {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            std::vector<Set::Scalar> row; Set::Scalar x;
            while (ls >> x) row.push_back(x);
            if (row.size() < 2) continue;
            if (value.m_ncol < 0) value.m_ncol = (int)row.size() - 1;
            if ((int)row.size() - 1 != value.m_ncol) Util::Abort(INFO, "IC::Profile1D: rows of different length in ", file);
            if (!value.m_s.empty() && !(row[0] > value.m_s.back())) Util::Abort(INFO, "IC::Profile1D: the first column must increase, ", file);
            value.m_s.push_back(row[0]);
            for (int c = 1; c <= value.m_ncol; ++c) value.m_v.push_back(row[c]);
        }
        if (value.m_s.size() < 2) Util::Abort(INFO, "IC::Profile1D: fewer than two rows in ", file);
        for (int c : value.m_columns)
            if (c > value.m_ncol) Util::Abort(INFO, "IC::Profile1D: column ", c, " requested, the file has ", value.m_ncol);
    }
private:
    std::vector<Set::Scalar> m_s, m_v;
    std::vector<int> m_columns;
    int m_ncol = 0, m_flip = 0;
    Set::Scalar m_x0 = 0.0;
};
}
#endif
