#ifndef INTEGRATOR_HYDRO2_SOLID_H
#define INTEGRATOR_HYDRO2_SOLID_H
#include <algorithm>
#include <cmath>
#include "Set/Base.H"
#include "BC/BC.H"
#include "IC/IC.H"
#include "IC/Constant.H"
#include "IC/Expression.H"
#include "BC/Constant.H"
#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
namespace Integrator {
// Hydro2_Solid.H -- everything Hydro2 needs for DEFORMING solids, so that Hydro2.cpp stays a
// fluid solver: parameters and parsing, the cobasis field and its boundary / initial
// conditions, the stress and elastic energy, the face states handed to the hllc_elastic
// Riemann solver, the cobasis transport row and the fluid-cell reset.  Hydro2 owns one
// ElasticSolid (`elastic`) and calls into it; nothing here runs unless elastic.on = 1.
// (The non-deforming embedded solid is EmbeddedSolid.H; ablation is Ablation.H.)
//
// DEFORMING (hyperelastic) SOLID PHASE for Hydro2 -- default OFF (elastic.on = 0).
//
// Model: N. Favrie, S. L. Gavrilyuk, R. Saurel, "Solid-fluid diffuse interface
// model in cases of extreme deformations", JCP 228 (2009) 6037-6077.  One of the
// two phases of the 6-equation model is a solid (or both are, sharing one
// deformation field -- e.g. an Al|Cu contact):
//   * hydrodynamic part: the phase's own stiffened-gas EOS (unchanged),
//   * elastic part: e^e = mu_s / (4 rho_s0) tr((Gt - I)^2),  Gt = G / |G|^(1/3)   (eqs. 9, 11)
//     with G = sum_b E^b (x) E^b the averaged Finger tensor built from the
//     averaged cobasis E^b = alpha_s^(1/3) grad X_b                             (eq. 22)
//   * cobasis transport  dE^b/dt + grad(E^b . v) = -rot E^b ^ v                 (eq. 21 / 39a)
//       i.e.  d_t E_i + v_k d_k E_i + E_k d_i v_k = 0
//   * mixture stress  sigma = -p I + S,  S = alpha_s dev(sigma_s)               (eq. 12, App. C)
//       S = -mu_s (alpha_s rho_s / rho_s0) dev(Gt^2 - Gt)
//     (App. C writes the prefactor as mu_s |G|^(1/2); |G|^(1/2) = alpha_s rho_s / rho_s0 by eq. 22.
//      The transported phase mass is used so a trace of solid in the fluid carries no stress.
//      With two solid phases the prefactor is sum_k mu_k alpha_k rho_k / rho_k0.)
//   * elastic energy per volume  W = (alpha_s rho_s / rho_s0) (mu_s / 4) tr((Gt - I)^2)   (eq. 46)
//     is part of the conserved total energy; the hydrodynamic internal energy is
//     rho E - kinetic - W.
// Small-strain check: E = I - grad(u)  ->  G ~ I - 2 eps,  S ~ 2 mu dev(eps)  (Hooke).
//
// Storage of the cobasis field (NEB components):
//   3D: comp 3 b + i = E^b_i.
//   2D (plane strain): E^1 = (c0, c1), E^2 = (c2, c3), E^3 = (0, 0, c4).  c4 is carried
//       because E^3_z = alpha_s^(1/3) is not 1 across a diffuse interface.
class ElasticSolid {
public:
    int          on = 0;                  // 1 = hyperelastic solid phase(s) on
    Set::Scalar  mu[2] = {0.0, 0.0};      // shear modulus of phase 0 (eta = 1) and phase 1 (eta = 0); 0 = fluid
    Set::Scalar  rho0[2] = {1.0, 1.0};    // reference (unstrained) density of each solid phase
    Set::Scalar  fluid_cut = 0.1;         // cells with a solid volume fraction below this are fluid: cobasis reset to isotropic (<= 0: never)
    Set::Scalar  reconstruct_cut = 0.99;  // ... only where the cell and both neighbours have a solid volume fraction above this
    int          objective_shear = 1;     // 1 = shear dissipation of the Riemann solver acts on the symmetric velocity gradient (no drag on rigid rotation)
    int          reconstruct = 1;         // 1 = limited (MC) linear reconstruction of the cobasis inside the solid (2nd order); 0 = zero slope (Favrie App. B)
    Set::Scalar  solid_cut = 0.5;         // a face transmits shear only if the solid volume fraction > solid_cut on both sides (Favrie Sec. 5.6)

    static constexpr int NEB = (AMREX_SPACEDIM == 2) ? 5 : 9;
    Set::Field<Set::Scalar> ebasis_mf, ebasis_old_mf;
    Set::Field<Set::Scalar> stress_mf;        // plot only: S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, W
    BC::BC<Set::Scalar>* bc = nullptr;
    // optional pre-strain: the solid's own cobasis e^b_i = dX_b/dx_i at t = 0 (components in the
    // storage order above; identity if absent).  det(e) should equal rho_s / rho_s0.
    IC::IC<Set::Scalar>* cobasis_ic = nullptr;

    ~ElasticSolid() { delete bc; delete cobasis_ic; }

    // cobasis components of one cell -> Em[b][i]
    template <class A>
    AMREX_GPU_HOST_DEVICE static void load(const A &eb, int i, int j, int k, Set::Scalar Em[3][3])
    {
#if AMREX_SPACEDIM == 2
        Em[0][0] = eb(i, j, k, 0); Em[0][1] = eb(i, j, k, 1); Em[0][2] = 0.0;
        Em[1][0] = eb(i, j, k, 2); Em[1][1] = eb(i, j, k, 3); Em[1][2] = 0.0;
        Em[2][0] = 0.0;            Em[2][1] = 0.0;            Em[2][2] = eb(i, j, k, 4);
#else
        for (int b = 0; b < 3; ++b)
            for (int d = 0; d < 3; ++d) Em[b][d] = eb(i, j, k, 3 * b + d);
#endif
    }
    // component index of E^b_i (2D: only b, i < 2, and (2,2))
    AMREX_GPU_HOST_DEVICE static int comp(int b, int i)
    {
#if AMREX_SPACEDIM == 2
        return (b == 2) ? 4 : 2 * b + i;
#else
        return 3 * b + i;
#endif
    }

    // monotonized-central limited slope from the backward (a) and forward (b) differences
    AMREX_GPU_HOST_DEVICE static Set::Scalar mc(Set::Scalar a, Set::Scalar b)
    {
        if (a * b <= 0.0) return 0.0;
        const Set::Scalar s = (a > 0.0) ? 1.0 : -1.0;
        return s * std::min(std::min(2.0 * std::abs(a), 2.0 * std::abs(b)), 0.5 * std::abs(a + b));
    }
    // mu-weighted reference volume fraction of solid: sum_k mu_k (alpha_k rho_k) / rho_k0
    AMREX_GPU_HOST_DEVICE Set::Scalar prefactor(Set::Scalar m0, Set::Scalar m1) const
    {
        return mu[0] * std::max(m0, 0.0) / rho0[0] + mu[1] * std::max(m1, 0.0) / rho0[1];
    }
    // reference volume fraction of solid (phases with mu > 0): sum_k (alpha_k rho_k) / rho_k0
    AMREX_GPU_HOST_DEVICE Set::Scalar refvol(Set::Scalar m0, Set::Scalar m1) const
    {
        return ((mu[0] > 0.0) ? std::max(m0, 0.0) / rho0[0] : 0.0) + ((mu[1] > 0.0) ? std::max(m1, 0.0) / rho0[1] : 0.0);
    }
    // volume fraction occupied by solid phases, from alpha of phase 0
    AMREX_GPU_HOST_DEVICE Set::Scalar solidfrac(Set::Scalar a0) const
    {
        return ((mu[0] > 0.0) ? a0 : 0.0) + ((mu[1] > 0.0) ? 1.0 - a0 : 0.0);
    }
    // S = sum_k alpha_k dev(sigma_k) and elastic energy per volume W from the cobasis.
    // m0, m1 = (alpha rho)_k, the transported phase masses.
    AMREX_GPU_HOST_DEVICE void stress(const Set::Scalar Em[3][3], Set::Scalar m0, Set::Scalar m1,
                                      Set::Scalar S[3][3], Set::Scalar &W) const
    {
        const Set::Scalar pre = prefactor(m0, m1);
        Set::Scalar G[3][3];
        for (int a = 0; a < 3; ++a)
            for (int c = 0; c < 3; ++c)
                G[a][c] = Em[0][a] * Em[0][c] + Em[1][a] * Em[1][c] + Em[2][a] * Em[2][c];
        const Set::Scalar detE = Em[0][0] * (Em[1][1] * Em[2][2] - Em[1][2] * Em[2][1])
                               - Em[0][1] * (Em[1][0] * Em[2][2] - Em[1][2] * Em[2][0])
                               + Em[0][2] * (Em[1][0] * Em[2][1] - Em[1][1] * Em[2][0]);
        const Set::Scalar detG = detE * detE;
        for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) S[a][c] = 0.0;
        W = 0.0;
        if (!(detG > 1.0e-300) || !(pre > 0.0)) return;
        const Set::Scalar s = 1.0 / std::cbrt(detG);
        Set::Scalar Gt[3][3], Gt2[3][3];
        for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) Gt[a][c] = s * G[a][c];
        for (int a = 0; a < 3; ++a)
            for (int c = 0; c < 3; ++c)
                Gt2[a][c] = Gt[a][0] * Gt[0][c] + Gt[a][1] * Gt[1][c] + Gt[a][2] * Gt[2][c];
        const Set::Scalar J1 = Gt[0][0] + Gt[1][1] + Gt[2][2];
        const Set::Scalar J2 = Gt2[0][0] + Gt2[1][1] + Gt2[2][2];
        for (int a = 0; a < 3; ++a)
            for (int c = 0; c < 3; ++c)
                S[a][c] = -pre * (Gt2[a][c] - Gt[a][c] - ((a == c) ? (J2 - J1) / 3.0 : 0.0));
        W = 0.25 * pre * (J2 - 2.0 * J1 + 3.0);
    }
    // elastic part of the squared longitudinal speed: sum_k Y_k (4/3) mu_k / rho_k = (4/3) sum_k mu_k alpha_k / rho
    AMREX_GPU_HOST_DEVICE Set::Scalar c2(Set::Scalar a0, Set::Scalar rho) const
    {
        return (4.0 / 3.0) * (mu[0] * a0 + mu[1] * (1.0 - a0)) / std::max(rho, 1.0e-300);
    }

    // ---------------------------------------------------------------------------------------
    // Input.  Called from Hydro2::Parse.
    // ---------------------------------------------------------------------------------------
    static void Parse(ElasticSolid &value, IO::ParmParse &pp, amrex::Vector<amrex::Geometry> &geom)
    {
        // [2026-10-05] DEFORMING SOLID PHASE (default off).  A phase with elastic.mu<k> > 0 of the
        // 6-equation model becomes a hyperelastic solid (Favrie, Gavrilyuk & Saurel, JCP 228,
        // 2009): its stiffened-gas EOS is the hydrodynamic part, elastic.mu<k> the shear modulus,
        // elastic.rhoref<k> the unstrained density.  Adds the cobasis field `ebasis` and the
        // hllc_elastic Riemann solver.  Not combined with the embedded rigid solid (solid.*) and
        // single-level only so far.
        pp_query_default("elastic.on", value.on, 0);
        if (!value.on) return;
        pp_query_default("elastic.mu0", value.mu[0], 0.0);        // shear modulus of phase 0 (eta = 1); 0 = fluid
        pp_query_default("elastic.mu1", value.mu[1], 0.0);        // shear modulus of phase 1 (eta = 0); 0 = fluid
        pp_query_default("elastic.rhoref0", value.rho0[0], 1.0);  // unstrained density of phase 0 (used if mu0 > 0)
        pp_query_default("elastic.rhoref1", value.rho0[1], 1.0);  // unstrained density of phase 1 (used if mu1 > 0)
        // optional pre-strained initial state: elastic.cobasis.ic.type = expression with
        // region0.. = e^1_x, e^1_y, e^2_x, e^2_y, e^3_z (2D) or the 9 components e^b_i (3D)
        if (pp.contains("elastic.cobasis.ic.type"))
            pp.select<IC::Constant,IC::Expression>("elastic.cobasis.ic", value.cobasis_ic, geom);
        pp_query_default("elastic.fluid_cut", value.fluid_cut, 0.1); // solid volume fraction below which a cell is fluid: its cobasis is reset to isotropic (<= 0 disables)
        pp_query_default("elastic.solid_cut", value.solid_cut, 0.5); // shear is transmitted across a face only if the solid volume fraction > this on both sides
        // 1 = second order: MC-limited linear reconstruction of the cobasis at faces where the cell
        // and both neighbours along the face normal are solid; 0 = zero slope everywhere (Favrie App. B)
        pp_query_default("elastic.reconstruct", value.reconstruct, 1);
        // 1 = the upwind shear dissipation at solid|solid faces acts on d(v)/dx + d(u)/dy instead of
        // d(v)/dx alone, so it does not brake a rigid rotation (see HLLC_Elastic.H); 0 = paper's 1D form
        pp_query_default("elastic.objective_shear", value.objective_shear, 1);
        // the reconstruction is per component, which is not rotation invariant where |E| varies
        // (E ~ alpha_s^(1/3) across the smeared surface): keep it to (nearly) pure solid cells
        pp_query_default("elastic.reconstruct_cut", value.reconstruct_cut, 0.99);

        // cobasis BC: zero-gradient on physical boundaries, periodic where the domain is
        // (BC::Constant does not look at the geometry, so say so explicitly)
        amrex::Vector<int> is_per(AMREX_SPACEDIM, 0);
        { amrex::ParmParse ppg("geometry"); ppg.queryarr("is_periodic", is_per, 0, AMREX_SPACEDIM); }
        amrex::Vector<std::string> el_bstr;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) el_bstr.push_back(is_per[d] ? "periodic" : "neumann");
        amrex::Vector<amrex::Real> el_z(NEB, 0.0);
        value.bc = new BC::Constant(NEB, el_bstr, el_bstr, AMREX_D_DECL(el_z, el_z, el_z), AMREX_D_DECL(el_z, el_z, el_z));
    }
    static std::vector<std::string> cobasisNames()
    {
#if AMREX_SPACEDIM == 2
        return { "1x", "1y", "2x", "2y", "3z" };
#else
        return { "1x", "1y", "1z", "2x", "2y", "2z", "3x", "3y", "3z" };
#endif
    }

    // ---------------------------------------------------------------------------------------
    // Field operations (one level).
    // ---------------------------------------------------------------------------------------
    // cobasis ghost cells: same-level / periodic exchange + the physical BC
    void FillGhosts(int lev, const amrex::Geometry &geom, Set::Scalar time)
    {
        bc->define(geom);
        ebasis_mf[lev]->FillBoundary(geom.periodicity());
        bc->FillBoundary(*ebasis_mf[lev], 0, ebasis_mf[lev]->nComp(), time, 0);
    }
    // Unstrained cobasis: E^b = (sum_solid alpha_s rho_s / rho_s0)^(1/3) e_b, i.e. an isotropic
    // <G> with rho_s0 |<G>|^(1/2) = alpha_s rho_s (Favrie 2009 eq. 22).  Only
    // <G>/|<G>|^(1/3) enters the stress, so the initial stress is zero everywhere -- unless a
    // pre-strain elastic.cobasis.ic is given, whose elastic energy is then added to rho E.
    void Init(int lev, const amrex::Geometry &geom, const amrex::MultiFab &m0_mf, const amrex::MultiFab &m1_mf,
              amrex::MultiFab &E_mf, amrex::MultiFab &Eold_mf)
    {
        if (cobasis_ic) cobasis_ic->Initialize(lev, ebasis_mf, 0.0);
        const bool have_ic = cobasis_ic != nullptr;
        for (amrex::MFIter mfi(*ebasis_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box bx = mfi.validbox();
            auto eb = ebasis_mf[lev]->array(mfi);
            auto m0 = m0_mf.const_array(mfi);
            auto m1 = m1_mf.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar rv = std::max(refvol(m0(i, j, k), m1(i, j, k)), 1.0e-12);
                if (have_ic)
                {
                    // E^b = alpha_s^(1/3) e^b with rho_s0 |<G>|^(1/2) = alpha_s rho_s:  E = e (refvol / det e)^(1/3)
                    Set::Scalar Em[3][3];
                    load(eb, i, j, k, Em);
                    const Set::Scalar det = Em[0][0] * (Em[1][1] * Em[2][2] - Em[1][2] * Em[2][1])
                                          - Em[0][1] * (Em[1][0] * Em[2][2] - Em[1][2] * Em[2][0])
                                          + Em[0][2] * (Em[1][0] * Em[2][1] - Em[1][1] * Em[2][0]);
                    if (det > 1.0e-12)
                    {
                        const Set::Scalar sc = std::cbrt(rv / det);
                        for (int n = 0; n < NEB; ++n) eb(i, j, k, n) *= sc;
                        return;
                    }
                }
                const Set::Scalar lam = std::cbrt(rv);
                for (int n = 0; n < NEB; ++n) eb(i, j, k, n) = 0.0;
                for (int b = 0; b < 3; ++b) eb(i, j, k, comp(b, b)) = lam;
            });
        }
        FillGhosts(lev, geom, 0.0);
        amrex::MultiFab::Copy(*ebasis_old_mf[lev], *ebasis_mf[lev], 0, 0, NEB, ebasis_mf[lev]->nGrow());
        // A pre-strained solid stores elastic energy: rho E (set in Mix from p and u) must carry it.
        if (have_ic)
            for (amrex::MFIter mfi(*ebasis_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box bx = mfi.validbox();
                auto eb = ebasis_mf[lev]->const_array(mfi);
                auto m0 = m0_mf.const_array(mfi);
                auto m1 = m1_mf.const_array(mfi);
                auto E  = E_mf.array(mfi);
                auto Eo = Eold_mf.array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    E(i, j, k) += energy(eb, i, j, k, m0(i, j, k), m1(i, j, k));  Eo(i, j, k) = E(i, j, k);
                });
            }
        stress_mf[lev]->setVal(0.0);
    }
    // FLUID CELLS HAVE NO DEFORMATION MEMORY.  The cobasis is transported everywhere, and in
    // the fluid it is stretched and rotated without bound by shear layers and vortices (measured
    // on the shock / cylinder test: |E| x25 and rotated by ~180 deg in the gas next to the
    // body); upwinding then feeds that garbage into the solid's surface cells (S and rho blow up
    // there, dt collapses).  Reset it to the unstrained isotropic value wherever the solid volume
    // fraction is below elastic.fluid_cut -- an instantaneous shear relaxation in the fluid.  W of
    // those cells (~mu alpha_s) returns to internal energy through the conserved rho E at the
    // next reinit.
    void ResetFluidCells(int lev, const amrex::MultiFab &eta_mf, const amrex::MultiFab &m0_mf, const amrex::MultiFab &m1_mf)
    {
        if (!(fluid_cut > 0.0)) return;
        const Set::Scalar fcut = fluid_cut;
        for (amrex::MFIter mfi(*ebasis_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box bx = mfi.validbox();
            auto eb = ebasis_mf[lev]->array(mfi);
            auto et = eta_mf.const_array(mfi);
            auto m0 = m0_mf.const_array(mfi);
            auto m1 = m1_mf.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar a0 = std::min(std::max(et(i, j, k), 0.0), 1.0);
                if (solidfrac(a0) >= fcut) return;
                const Set::Scalar lam = std::cbrt(std::max(refvol(m0(i, j, k), m1(i, j, k)), 1.0e-12));
                for (int n = 0; n < NEB; ++n) eb(i, j, k, n) = 0.0;
                for (int b = 0; b < 3; ++b) eb(i, j, k, comp(b, b)) = lam;
            });
        }
    }

    // ---------------------------------------------------------------------------------------
    // Per-cell / per-face kernels, called inside Hydro2's loops.
    // ---------------------------------------------------------------------------------------
    // elastic energy per volume of one cell
    template <class A>
    AMREX_GPU_HOST_DEVICE Set::Scalar energy(const A &eb, int i, int j, int k, Set::Scalar m0, Set::Scalar m1) const
    {
        Set::Scalar Em[3][3], S[3][3], W;
        load(eb, i, j, k, Em);
        stress(Em, m0, m1, S, W);
        return W;
    }
    template <class A>
    AMREX_GPU_HOST_DEVICE bool isSolid(const A &eta, int i, int j, int k, Set::Scalar cut) const
    {
        return solidfrac(std::min(std::max(eta(i, j, k), 0.0), 1.0)) > cut;
    }
    AMREX_GPU_HOST_DEVICE static void emFrom(const Set::Scalar Ef[NEB], Set::Scalar Em[3][3])
    {
#if AMREX_SPACEDIM == 2
        Em[0][0] = Ef[0]; Em[0][1] = Ef[1]; Em[0][2] = 0.0;
        Em[1][0] = Ef[2]; Em[1][1] = Ef[3]; Em[1][2] = 0.0;
        Em[2][0] = 0.0;   Em[2][1] = 0.0;   Em[2][2] = Ef[4];
#else
        for (int b2 = 0; b2 < 3; ++b2) for (int d2 = 0; d2 < 3; ++d2) Em[b2][d2] = Ef[3 * b2 + d2];
#endif
    }
    // Cobasis at the face of cell (ci,cj,ck) on side sd (+1 hi, -1 lo) along dir.  Linear, MC
    // limited, only where this cell and both neighbours along dir are (nearly pure) solid; else
    // the cell value -- the paper's zero slope (Favrie 2009, App. B), which alone makes shear and
    // rotation first-order dissipative.
    template <class A, class B>
    AMREX_GPU_HOST_DEVICE void faceCobasis(const A &eb, const B &eta, int ci, int cj, int ck, int dir, int sd, Set::Scalar Ef[NEB]) const
    {
        for (int n = 0; n < NEB; ++n) Ef[n] = eb(ci, cj, ck, n);
        if (!reconstruct) return;
        const int a = (dir == 0), b = (dir == 1), c = (dir == 2);
        if (!(isSolid(eta, ci, cj, ck, reconstruct_cut) && isSolid(eta, ci - a, cj - b, ck - c, reconstruct_cut)
              && isSolid(eta, ci + a, cj + b, ck + c, reconstruct_cut))) return;
        for (int n = 0; n < NEB; ++n)
            Ef[n] += 0.5 * sd * mc(eb(ci, cj, ck, n) - eb(ci - a, cj - b, ck - c, n),
                                   eb(ci + a, cj + b, ck + c, n) - eb(ci, cj, ck, n));
    }
    // Solid part of the Riemann state on one side of a face: stress from the face cobasis,
    // elastic sound-speed term, solid flag.
    template <class A, class B, class C>
    AMREX_GPU_HOST_DEVICE void faceState(Solver::Local::FluidRiemann::State &st, const A &eb, const B &eta,
                                         const C &m0, const C &m1, int ci, int cj, int ck, int dir, int sd) const
    {
        Set::Scalar Em[3][3], S[3][3], W, Ef[NEB];
        faceCobasis(eb, eta, ci, cj, ck, dir, sd, Ef);
        emFrom(Ef, Em);
        stress(Em, m0(ci, cj, ck), m1(ci, cj, ck), S, W);
        const int t1 = (dir + 1) % AMREX_SPACEDIM;
        st.S_nn = S[dir][dir];
        st.S_nt = S[dir][t1];
#if AMREX_SPACEDIM == 3
        st.S_nt2 = S[dir][(dir + 2) % AMREX_SPACEDIM];
#endif
        const Set::Scalar a0 = std::min(std::max(eta(ci, cj, ck), 0.0), 1.0);
        st.c2_el = c2(a0, m0(ci, cj, ck) + m1(ci, cj, ck));
        st.solid = (solidfrac(a0) > solid_cut) ? 1 : 0;
    }
    // OBJECTIVE SHEAR DISSIPATION input for hllc_elastic (State::dvt): dx_n d(u_n)/d(x_t) at the
    // face = mean over the two cells of the transverse difference of the normal velocity, taken
    // through solid neighbours only; un(i,j,k) returns the normal velocity of a cell.
    // LIMITED against the reconstructed tangential jump J = v_R - v_L: the correction c may only
    // cancel part of J (J + c stays between 0 and J), so the net shear dissipation is never
    // negative and never larger than the 1D solver's.  Rigid rotation at first order: c = -J, no
    // drag.  Smooth second-order regions: J ~ 0, so c ~ 0.  (Unlimited, it was anti-dissipative
    // for oblique shear waves and a spinning disc spun up.)
    template <class B, class F>
    AMREX_GPU_HOST_DEVICE Set::Scalar shearCorrection(const B &eta, F &&un, int lo_i, int lo_j, int lo_k,
                                                      int hi_i, int hi_j, int hi_k, int t, Set::Scalar dxn, Set::Scalar dxt, Set::Scalar J) const
    {
        auto dtrans = [&](int ci, int cj, int ck) -> Set::Scalar {
            const int a = (t == 0), b = (t == 1), c = (t == 2);
            const bool sp = isSolid(eta, ci + a, cj + b, ck + c, solid_cut), sm = isSolid(eta, ci - a, cj - b, ck - c, solid_cut);
            const Set::Scalar uc = un(ci, cj, ck);
            const Set::Scalar up = sp ? un(ci + a, cj + b, ck + c) : uc;
            const Set::Scalar um = sm ? un(ci - a, cj - b, ck - c) : uc;
            const int nn = (sp ? 1 : 0) + (sm ? 1 : 0);
            return (nn > 0) ? (up - um) / nn : 0.0;       // difference per ONE transverse spacing
        };
        const Set::Scalar c = 0.5 * (dtrans(lo_i, lo_j, lo_k) + dtrans(hi_i, hi_j, hi_k)) * dxn / dxt;
        return (J > 0.0) ? std::min(0.0, std::max(c, -J)) : std::max(0.0, std::min(c, -J));
    }
    // COBASIS ROW  (Favrie 2009 eq. 39a)   d_t E^b_i + v_k d_k E^b_i + E^b_k d_i v_k = 0
    // written as  -div(E_i u*) + E_i div(u*) - E_k d_i v*_k  and discretised like the alpha row
    // (Favrie eq. 60): E upwinded on the contact speed u* (with the face reconstruction above),
    // velocity differences from the Riemann star velocities on this cell's side of each face.  In
    // direction i the E_i row is then conservative, d_t E_i + d_i(E_i u) + E_t d_i v_t = 0.
    // Fl[d], Fh[d]: the Riemann fluxes at the lo / hi face of the cell in direction d.
    // Also fills the plot diagnostics (S = alpha_s dev(sigma_s), W) when diag is valid.
    template <class A, class B, class C, class R, class FluxT>
    AMREX_GPU_HOST_DEVICE void cobasisRHS(const R &rhs, const R &diag, const A &eb, const B &eta, const C &m0, const C &m1,
                                          const FluxT *const Fl[AMREX_SPACEDIM], const FluxT *const Fh[AMREX_SPACEDIM],
                                          const Set::Scalar *DX, int i, int j, int k) const
    {
        Set::Scalar dv[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};   // dv[i][k] = d_i v_k
        Set::Scalar divu_el = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            dv[d][d] = (Fh[d]->u_interface - Fl[d]->u_interface) / DX[d];
            dv[d][(d + 1) % AMREX_SPACEDIM] = (Fh[d]->vt_L - Fl[d]->vt_R) / DX[d];
#if AMREX_SPACEDIM == 3
            dv[d][(d + 2) % AMREX_SPACEDIM] = (Fh[d]->wt_L - Fl[d]->wt_R) / DX[d];
#endif
            divu_el += dv[d][d];
        }
        Set::Scalar adv[NEB];
        for (int n = 0; n < NEB; ++n) adv[n] = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            const int di = (d == 0), dj = (d == 1), dk = (d == 2);
            Set::Scalar ehi[NEB], elo[NEB];
            if (Fh[d]->u_interface > 0.0) faceCobasis(eb, eta, i, j, k, d, +1, ehi); else faceCobasis(eb, eta, i + di, j + dj, k + dk, d, -1, ehi);
            if (Fl[d]->u_interface > 0.0) faceCobasis(eb, eta, i - di, j - dj, k - dk, d, +1, elo); else faceCobasis(eb, eta, i, j, k, d, -1, elo);
            for (int n = 0; n < NEB; ++n)
                adv[n] += (Fh[d]->u_interface * ehi[n] - Fl[d]->u_interface * elo[n]) / DX[d];
        }
        for (int n = 0; n < NEB; ++n)
            rhs(i, j, k, n) = -adv[n] + eb(i, j, k, n) * divu_el;
        for (int b = 0; b < AMREX_SPACEDIM; ++b)
            for (int c = 0; c < AMREX_SPACEDIM; ++c)
            {
                Set::Scalar src = 0.0;
                for (int kk = 0; kk < AMREX_SPACEDIM; ++kk)
                    src += eb(i, j, k, comp(b, kk)) * dv[c][kk];
                rhs(i, j, k, comp(b, c)) -= src;
            }
        Set::Scalar Em[3][3], S[3][3], W;
        load(eb, i, j, k, Em);
        stress(Em, m0(i, j, k), m1(i, j, k), S, W);
        diag(i, j, k, 0) = S[0][0]; diag(i, j, k, 1) = S[1][1]; diag(i, j, k, 2) = S[2][2];
        diag(i, j, k, 3) = S[0][1]; diag(i, j, k, 4) = S[0][2]; diag(i, j, k, 5) = S[1][2];
        diag(i, j, k, 6) = W;
    }
};
}
#endif
