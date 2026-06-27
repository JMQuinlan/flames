//
// Self-contained STL triangle-mesh reader + signed-distance query (Route B).
//
// Reads an STL file (binary OR ASCII), builds a bounding-volume hierarchy (BVH),
// and answers SignedDistance(x,y,z):
//     d < 0   INSIDE  the mesh (solid)
//     d > 0   OUTSIDE the mesh (fluid)
// |d| = distance to the nearest triangle (BVH-accelerated, O(log N) per query).
// The sign is by RAY-PARITY: cast a few rays and count triangle crossings
// (odd = inside); majority-vote of 3 off-axis rays for robustness on watertight
// CAD meshes.  This needs NO AMReX embedded-boundary module (USE_EB) -- it is
// fully self-contained, so the diffuse-interface solver builds without EB.
//
// Used by IC::STL to set the embedded-solid indicator phi = 0.5*(1+tanh(d/eps)).
//

#ifndef UTIL_STL_H
#define UTIL_STL_H

#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "Util/Util.H"

namespace Util
{
namespace detail_stl
{
    struct V3 { double x, y, z; };
    inline V3   operator-(V3 a, V3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
    inline V3   operator+(V3 a, V3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
    inline V3   operator*(V3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
    inline double dot(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
    inline V3   cross(V3 a, V3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
    inline double len2(V3 a) { return dot(a, a); }

    // Closest point on triangle (a,b,c) to p (Ericson, Real-Time Collision Detection).
    inline V3 ClosestOnTri(V3 p, V3 a, V3 b, V3 c)
    {
        V3 ab = b-a, ac = c-a, ap = p-a;
        double d1 = dot(ab, ap), d2 = dot(ac, ap);
        if (d1 <= 0 && d2 <= 0) return a;
        V3 bp = p-b; double d3 = dot(ab, bp), d4 = dot(ac, bp);
        if (d3 >= 0 && d4 <= d3) return b;
        double vc = d1*d4 - d3*d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) { double v = d1/(d1-d3); return a + ab*v; }
        V3 cp = p-c; double d5 = dot(ab, cp), d6 = dot(ac, cp);
        if (d6 >= 0 && d5 <= d6) return c;
        double vb = d5*d2 - d1*d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) { double w = d2/(d2-d6); return a + ac*w; }
        double va = d3*d6 - d5*d4;
        if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) { double w = (d4-d3)/((d4-d3)+(d5-d6)); return b + (c-b)*w; }
        double den = 1.0/(va+vb+vc), v = vb*den, w = vc*den;
        return a + ab*v + ac*w;
    }

    // Moller-Trumbore: does the ray (orig + t*dir, t>eps) cross triangle (a,b,c)?
    inline bool RayHitsTri(V3 orig, V3 dir, V3 a, V3 b, V3 c)
    {
        const double EPS = 1e-12;
        V3 e1 = b-a, e2 = c-a, pv = cross(dir, e2);
        double det = dot(e1, pv);
        if (det > -EPS && det < EPS) return false;          // parallel
        double inv = 1.0/det;
        V3 tv = orig - a;
        double u = dot(tv, pv) * inv;
        if (u < 0.0 || u > 1.0) return false;
        V3 qv = cross(tv, e1);
        double v = dot(dir, qv) * inv;
        if (v < 0.0 || u+v > 1.0) return false;
        double t = dot(e2, qv) * inv;
        return t > EPS;
    }
}

class STL
{
public:
    using V3 = detail_stl::V3;

    // Read + place the mesh: vertex -> vertex*scale + center.  reverse_normal!=0
    // swaps inside<->outside.  Builds the BVH up front.
    void Read(const std::string &fname, double scale, std::array<double,3> center, int reverse_normal)
    {
        m_sign = reverse_normal ? -1.0 : 1.0;
        ReadFile(fname, scale, center);
        Util::Message(INFO, "Util::STL: read ", m_tris.size(), " triangles from ", fname);
        if (m_tris.empty()) Util::Abort(INFO, "Util::STL: no triangles read from ", fname);
        m_order.resize(m_tris.size());
        for (size_t i = 0; i < m_order.size(); ++i) m_order[i] = (int)i;
        m_nodes.reserve(2*m_tris.size());
        BuildNode(0, (int)m_tris.size());
    }

    // Signed distance to the surface.  <0 inside (solid), >0 outside (fluid).
    double SignedDistance(double x, double y, double z) const
    {
        V3 p{x, y, z};
        double d = std::sqrt(Nearest2(p));
        // ray-parity sign, majority vote over 3 off-axis directions
        static const V3 dirs[3] = {{1.0, 0.013, 0.017}, {0.011, 1.0, 0.007}, {0.009, 0.015, 1.0}};
        int inside_votes = 0;
        for (int r = 0; r < 3; ++r)
            if (CountRayHits(p, dirs[r]) & 1) ++inside_votes;
        double s = (inside_votes >= 2) ? -1.0 : 1.0;   // inside -> negative
        return m_sign * s * d;
    }

private:
    struct Tri { V3 v0, v1, v2; };
    struct Node { V3 bmin, bmax; int left = -1, right = -1; int start = 0, count = 0; };

    std::vector<Tri>  m_tris;
    std::vector<int>  m_order;     // triangle indices, permuted by the BVH build
    std::vector<Node> m_nodes;     // node 0 = root
    double m_sign = 1.0;
    static constexpr int LEAF = 4;

    // ---- file reading (binary or ASCII auto-detect) ----
    void ReadFile(const std::string &fname, double scale, std::array<double,3> c)
    {
        std::ifstream f(fname, std::ios::binary | std::ios::ate);
        if (!f) Util::Abort(INFO, "Util::STL: cannot open ", fname);
        std::streamsize sz = f.tellg();
        f.seekg(0);
        bool binary = false;
        if (sz >= 84)
        {
            char hdr[80]; f.read(hdr, 80);
            uint32_t ntri = 0; f.read(reinterpret_cast<char*>(&ntri), 4);
            if ((std::streamsize)(84 + 50ull*ntri) == sz) binary = true;   // exact-size => binary
            f.seekg(0);
        }
        if (binary) ReadBinary(f, scale, c);
        else        ReadAscii(fname, scale, c);
    }

    void ReadBinary(std::ifstream &f, double s, std::array<double,3> c)
    {
        char hdr[80]; f.read(hdr, 80);
        uint32_t ntri = 0; f.read(reinterpret_cast<char*>(&ntri), 4);
        m_tris.reserve(ntri);
        for (uint32_t i = 0; i < ntri; ++i)
        {
            float buf[12]; uint16_t attr;
            f.read(reinterpret_cast<char*>(buf), 48);     // normal(3) + 3 verts(9)
            f.read(reinterpret_cast<char*>(&attr), 2);
            Tri t;
            t.v0 = {buf[3]*s+c[0], buf[4]*s+c[1], buf[5]*s+c[2]};
            t.v1 = {buf[6]*s+c[0], buf[7]*s+c[1], buf[8]*s+c[2]};
            t.v2 = {buf[9]*s+c[0], buf[10]*s+c[1], buf[11]*s+c[2]};
            m_tris.push_back(t);
        }
    }

    void ReadAscii(const std::string &fname, double s, std::array<double,3> c)
    {
        std::ifstream f(fname);
        std::string tok;
        std::vector<V3> v;
        while (f >> tok)
        {
            if (tok == "vertex")
            {
                double x, y, z; f >> x >> y >> z;
                v.push_back({x*s+c[0], y*s+c[1], z*s+c[2]});
                if (v.size() == 3) { m_tris.push_back({v[0], v[1], v[2]}); v.clear(); }
            }
        }
    }

    // ---- BVH build (median split on the longest centroid axis) ----
    int BuildNode(int start, int count)
    {
        int idx = (int)m_nodes.size();
        m_nodes.push_back(Node());
        // bounding box of this range
        V3 bmin{ 1e300, 1e300, 1e300}, bmax{-1e300,-1e300,-1e300};
        for (int i = start; i < start+count; ++i)
        {
            const Tri &t = m_tris[m_order[i]];
            for (const V3 &p : {t.v0, t.v1, t.v2})
            {
                bmin = {std::min(bmin.x,p.x), std::min(bmin.y,p.y), std::min(bmin.z,p.z)};
                bmax = {std::max(bmax.x,p.x), std::max(bmax.y,p.y), std::max(bmax.z,p.z)};
            }
        }
        m_nodes[idx].bmin = bmin; m_nodes[idx].bmax = bmax;

        if (count <= LEAF) { m_nodes[idx].start = start; m_nodes[idx].count = count; return idx; }

        // split on the longest axis of the centroid bounds
        V3 cmin{ 1e300, 1e300, 1e300}, cmax{-1e300,-1e300,-1e300};
        auto cen = [&](int i){ const Tri &t = m_tris[m_order[i]];
            return V3{(t.v0.x+t.v1.x+t.v2.x)/3, (t.v0.y+t.v1.y+t.v2.y)/3, (t.v0.z+t.v1.z+t.v2.z)/3}; };
        for (int i = start; i < start+count; ++i)
        {
            V3 ct = cen(i);
            cmin = {std::min(cmin.x,ct.x), std::min(cmin.y,ct.y), std::min(cmin.z,ct.z)};
            cmax = {std::max(cmax.x,ct.x), std::max(cmax.y,ct.y), std::max(cmax.z,ct.z)};
        }
        V3 ext = cmax - cmin;
        int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
        int mid = start + count/2;
        std::nth_element(m_order.begin()+start, m_order.begin()+mid, m_order.begin()+start+count,
            [&](int a, int b){
                const Tri &ta = m_tris[a], &tb = m_tris[b];
                double ca = (axis==0)?(ta.v0.x+ta.v1.x+ta.v2.x):(axis==1)?(ta.v0.y+ta.v1.y+ta.v2.y):(ta.v0.z+ta.v1.z+ta.v2.z);
                double cb = (axis==0)?(tb.v0.x+tb.v1.x+tb.v2.x):(axis==1)?(tb.v0.y+tb.v1.y+tb.v2.y):(tb.v0.z+tb.v1.z+tb.v2.z);
                return ca < cb;
            });
        int l = BuildNode(start, mid-start);
        int r = BuildNode(mid, start+count-mid);
        m_nodes[idx].left = l; m_nodes[idx].right = r; m_nodes[idx].count = 0;
        return idx;
    }

    // squared distance from p to an AABB (0 if inside) -- for BVH pruning
    static double BoxDist2(V3 p, V3 lo, V3 hi)
    {
        double dx = std::max({lo.x-p.x, 0.0, p.x-hi.x});
        double dy = std::max({lo.y-p.y, 0.0, p.y-hi.y});
        double dz = std::max({lo.z-p.z, 0.0, p.z-hi.z});
        return dx*dx + dy*dy + dz*dz;
    }

    // nearest squared distance via recursive BVH traversal with pruning
    double Nearest2(V3 p) const
    {
        double best = std::numeric_limits<double>::max();
        Nearest2Rec(0, p, best);
        return best;
    }
    void Nearest2Rec(int ni, V3 p, double &best) const
    {
        const Node &n = m_nodes[ni];
        if (BoxDist2(p, n.bmin, n.bmax) > best) return;       // prune
        if (n.count > 0)                                       // leaf
        {
            for (int i = n.start; i < n.start+n.count; ++i)
            {
                const Tri &t = m_tris[m_order[i]];
                V3 cp = detail_stl::ClosestOnTri(p, t.v0, t.v1, t.v2);
                double d2 = detail_stl::len2(p - cp);
                if (d2 < best) best = d2;
            }
            return;
        }
        // visit nearer child first
        double dl = BoxDist2(p, m_nodes[n.left].bmin,  m_nodes[n.left].bmax);
        double dr = BoxDist2(p, m_nodes[n.right].bmin, m_nodes[n.right].bmax);
        if (dl <= dr) { Nearest2Rec(n.left, p, best);  Nearest2Rec(n.right, p, best); }
        else          { Nearest2Rec(n.right, p, best); Nearest2Rec(n.left,  p, best); }
    }

    // does the ray (p + t*dir) intersect this node's AABB for some t>0?
    static bool RayHitsBox(V3 p, V3 dir, V3 lo, V3 hi)
    {
        double tmin = 0.0, tmax = std::numeric_limits<double>::max();
        for (int a = 0; a < 3; ++a)
        {
            double o = (a==0)?p.x:(a==1)?p.y:p.z;
            double d = (a==0)?dir.x:(a==1)?dir.y:dir.z;
            double l = (a==0)?lo.x:(a==1)?lo.y:lo.z;
            double h = (a==0)?hi.x:(a==1)?hi.y:hi.z;
            if (std::fabs(d) < 1e-300) { if (o < l || o > h) return false; }
            else
            {
                double inv = 1.0/d, t1 = (l-o)*inv, t2 = (h-o)*inv;
                if (t1 > t2) std::swap(t1, t2);
                tmin = std::max(tmin, t1); tmax = std::min(tmax, t2);
                if (tmin > tmax) return false;
            }
        }
        return true;
    }

    int CountRayHits(V3 p, V3 dir) const
    {
        int count = 0; CountRayHitsRec(0, p, dir, count); return count;
    }
    void CountRayHitsRec(int ni, V3 p, V3 dir, int &count) const
    {
        const Node &n = m_nodes[ni];
        if (!RayHitsBox(p, dir, n.bmin, n.bmax)) return;
        if (n.count > 0)
        {
            for (int i = n.start; i < n.start+n.count; ++i)
            {
                const Tri &t = m_tris[m_order[i]];
                if (detail_stl::RayHitsTri(p, dir, t.v0, t.v1, t.v2)) ++count;
            }
            return;
        }
        CountRayHitsRec(n.left,  p, dir, count);
        CountRayHitsRec(n.right, p, dir, count);
    }
};
}
#endif
