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
    // [2026-10-06] SEVERAL SOLIDS IN ONE PHASE (species stage 2b).  When one phase carries solid SPECIES
    // (species.eos = mixture with species.phaseK.pinf / rho_ref) and elastic.species.mu is given, the material
    // constants of that phase are evaluated per cell from the species partial densities m_i, weighted with the
    // reference volumes v_i = m_i / rho_ref,i:
    //     1/rho0 = sum v_i / sum m_i,   mu = sum mu_i v_i / sum v_i,   sigma_y, K_y likewise
    // so that  mu m/rho0 = sum mu_i m_i/rho_ref,i  (the prefactor of the stress) is exact.  Mat is that set of
    // constants for one cell; with no solid species it is the per-phase input and the code is unchanged.
    struct Mat { Set::Scalar mu[2], rho0[2], sigma_y[2], Ky[2], chi[2], gtn[2]; };
    // [2026-10-06] COMPACTION ENERGY AND GTN YIELD FUNCTION (elastic.compaction = 1, default 0 = bit-identical).
    // Favrie & Schmidmayer, CMAME 455 (2026) 118880, Secs. 2 - 3, for ONE solid phase (phase 0, eta = alpha_s) and
    // a fluid (phase 1) in its pores.  The paper's configuration variable lambda^e enters only through
    //     xi = rho_r lambda^e / rho_s,   and   alpha_ref := alpha_s / xi   obeys   d(alpha_ref)/dt = -alpha_ref r'
    // (material derivative; from their eqs. 3 and 16).  alpha_ref -- the plastically accommodated solid fraction --
    // is carried by the solver's advected colour function `cfun` (same discretisation as the alpha row), which starts
    // equal to alpha_s (xi = 1) and is changed only by the plastic source.
    //   compaction energy per unit solid mass (eqs. 41, 54):  kappa = (chi/rho_r) f(alpha_s) |xi - 1|^n,
    //        f = alpha_s^n_kappa / (1 + a_kappa (1 - alpha_s))
    //        [ASSUMPTION: the paper gives chi in Pa and kappa in J/kg; the 1/rho_r is mine, cf. their eq. 53]
    //   energy per volume  W_c = (alpha_s rho_s/rho_r) chi f |xi - 1|^n                (added to the elastic energy)
    //   pi_s = rho_r xi dkappa/dxi  (eq. 27);   P_c = rho_s xi dkappa/dxi + alpha_s rho_s dkappa/dalpha_s
    //   equilibrium (eq. 11):  p_fluid = p_solid - P_c.  Realised WITHOUT touching the relaxation solver: the solid's
    //   stiffness is shifted per cell, pi_hat = pi + P_c/gamma, so the solver's "solid pressure" is the effective
    //   pressure p_s - P_c = -A_s (eq. 24) and its relaxation equalises A_s and A_f; the true -alpha_s p_s of the
    //   mixture stress (eq. 9) is restored by adding -alpha_s P_c to the normal stress of the face states.
    //   P_c is evaluated from the state at the start of a stage (lagged, like the species constants).
    // GTN yield function (eq. 60), porosity = 1 - alpha_s:
    //     g = sigma_y [ (sigma_eq/sigma_y)^2 + 2 q1 (1 - alpha_s) cosh(3 q2 rho_s pi_s/(2 sigma_y rho_r)) - 1 - q3 (1 - alpha_s)^2 ]
    //   Perzyna rate gamma' = <g>/K_y (eq. 64, N_y = 1).  FLOW RULE: the paper's Sec. 2.2.4 is EMPTY in the available
    //   version; the associated rule of the generalised-standard-material framework is used here,
    //     R' = gamma' dg/dtau (deviatoric part 3 S/sigma_y, eq. 62),    r' = -gamma' dg/dpi_s
    //   (the sign that makes -pi_s r' >= 0), so alpha_ref <- alpha_ref exp(dgamma dg/dpi_s).  The spherical part of
    //   R' is not applied to the cobasis: only its isochoric part enters the stress here.
    // NOT implemented: finite-rate pressure relaxation (the paper gives no value of its coefficient omega), the
    // porosity dependence of the shear modulus (the paper also sets mu_alpha = mu_s), more than one solid phase.
    int          comp_on = 0;
    Set::Scalar  chi[2] = {0.0, 0.0};     // compaction modulus of phase 0 / 1 (only phase 0 may be the solid)
    int          gtn[2] = {0, 0};         // 1 = GTN yield function for that phase (else von Mises)
    Set::Scalar  comp_n = 2.0, comp_nk = 0.5, comp_ak = 1.8e5;      // n_k, n_kappa, a_kappa
    Set::Scalar  q1 = 1.0, q2 = 1.0, q3 = 1.0;
    Set::Scalar  sp_chi[24] = {}, sp_gtn[24] = {};
    struct Comp { Set::Scalar W, Pc, pis; };
    // compaction state of a cell.  as = solid volume fraction, m0 = alpha_s rho_s, aref = cfun
    AMREX_GPU_HOST_DEVICE Comp compaction(const Mat &M, Set::Scalar as, Set::Scalar m0, Set::Scalar aref) const
    {
        Comp c; c.W = 0.0; c.Pc = 0.0; c.pis = 0.0;
        if (!comp_on || !(M.chi[0] > 0.0) || !(as > 1.0e-6) || !(aref > 1.0e-6) || !(m0 > 0.0)) return c;
        as = std::min(as, 1.0);
        const Set::Scalar xi = as / aref, d = xi - 1.0, ad = std::abs(d), sg = (d >= 0.0) ? 1.0 : -1.0;
        const Set::Scalar den = 1.0 + comp_ak * (1.0 - as), an = std::pow(as, comp_nk);
        const Set::Scalar f = an / den, fp = comp_nk * an / (as * den) + an * comp_ak / (den * den);
        const Set::Scalar pw = std::pow(ad, comp_n), pw1 = (ad > 0.0) ? sg * comp_n * std::pow(ad, comp_n - 1.0) : 0.0;
        const Set::Scalar rv = m0 / M.rho0[0];                     // alpha_s rho_s / rho_r
        c.W = rv * M.chi[0] * f * pw;
        c.pis = M.chi[0] * f * xi * pw1;
        c.Pc = (rv / as) * c.pis + rv * M.chi[0] * fp * pw;
        return c;
    }
    static constexpr int SPMAX = 24;
    int          sp_on = 0, sp_ph = 0, sp_c0 = 0, sp_n = 0;
    Set::Scalar  sp_mu[SPMAX] = {}, sp_r0[SPMAX] = {}, sp_sy[SPMAX] = {}, sp_ky[SPMAX] = {};
    Set::Scalar  fluid_cut = 0.1;         // cells with a solid volume fraction below this are fluid: cobasis reset to isotropic (<= 0: never)
    Set::Scalar  reconstruct_cut = 0.99;  // ... only where the cell and both neighbours have a solid volume fraction above this
    int          objective_shear = 1;     // 1 = shear dissipation of the Riemann solver acts on the symmetric velocity gradient (no drag on rigid rotation)
    int          reconstruct = 1;         // 1 = limited (MC) linear reconstruction of the cobasis inside the solid (2nd order); 0 = zero slope (Favrie App. B)
    Set::Scalar  solid_cut = 0.5;         // a face transmits shear only if the solid volume fraction > solid_cut on both sides (Favrie Sec. 5.6)
    // [2026-10-06] ELASTO-VISCOPLASTICITY (elastic.plastic = 1, default 0 = hyperelastic, bit-identical).
    // Favrie & Schmidmayer, CMAME 455 (2026) 118880: the elastic deformation relaxes when the yield function is
    // exceeded,  dF_e^-1/dt + F_e^-1 grad u = F_e^-1 R',  R' = gamma' dg/dtau  (their eqs. 18, 62-64).  Implemented:
    // von Mises  g = sigma_eq - sigma_y  (eq. 61),  dg/dtau = 3 S/(2 sigma_eq)  (eq. 63),  Perzyna rate
    // gamma' = <g>/K_y  (eq. 64 with N_y = 1).  The GTN yield function and the compaction energy are a separate option
    // (elastic.compaction, struct Comp above); several solids go in as species (struct Mat).  NOT implemented: the
    // finite-rate pressure relaxation of that paper.
    int          plastic = 0;
    Set::Scalar  sigma_y[2] = {0.0, 0.0}; // yield stress of phase 0 / 1 (<= 0: that phase never yields)
    Set::Scalar  Ky[2] = {0.0, 0.0};      // plastic viscosity K_y [stress x time]; 0 = rate-independent (perfect plasticity)
    Set::Field<Set::Scalar> plastic_mf;   // plot: plastic strain rate of the last step, von Mises stress of the solid

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
    // material constants of the two phases (no solid species: the inputs)
    AMREX_GPU_HOST_DEVICE Mat baseMat() const
    {
        Mat M;
        for (int p = 0; p < 2; ++p) { M.mu[p] = mu[p]; M.rho0[p] = rho0[p]; M.sigma_y[p] = sigma_y[p]; M.Ky[p] = Ky[p]; M.chi[p] = chi[p]; M.gtn[p] = gtn[p]; }
        return M;
    }
    // ... of one cell: the phase that carries solid species gets the reference-volume weighted constants
    template <class SP>
    AMREX_GPU_HOST_DEVICE Mat matAt(const SP &sp, int i, int j, int k) const
    {
        Mat M = baseMat();
        if (!sp_on) return M;
        // [2026-10-10] The yield stress and the plastic viscosity are averaged over the species that CARRY SHEAR
        // (mu_i > 0) only.  A fluid species of the phase (mu_i = 0, e.g. an explosive next to its confiner) has no
        // deviatoric stress and no yield surface: counted as "never yields" (1e300) it switched plasticity off in every
        // cell that held any trace of it -- the copper of the explosive-filled vessel carried 7 GPa of von Mises
        // stress against a yield stress of 0.07 GPa.  With all species solid the average is unchanged.
        Set::Scalar sm = 0.0, sv = 0.0, vs = 0.0, a = 0.0, b = 0.0, c = 0.0, e = 0.0, h = 0.0;
        for (int s = 0; s < sp_n; ++s)
        {
            const Set::Scalar mi = std::max(sp(i, j, k, sp_c0 + s), 0.0), vi = mi / sp_r0[s];
            sm += mi; sv += vi; a += vi * sp_mu[s];
            if (sp_mu[s] > 0.0) { vs += vi; b += vi * ((sp_sy[s] > 0.0) ? sp_sy[s] : 1.0e300); c += vi * sp_ky[s]; }
            e += vi * sp_chi[s]; h += vi * sp_gtn[s];
        }
        if (!(sv > 1.0e-300)) return M;
        M.mu[sp_ph] = a / sv; M.rho0[sp_ph] = sm / sv; M.Ky[sp_ph] = (vs > 1.0e-300) ? c / vs : 0.0;
        M.chi[sp_ph] = e / sv; M.gtn[sp_ph] = h / sv;
        M.sigma_y[sp_ph] = (vs > 1.0e-300 && b / vs < 1.0e299) ? b / vs : 0.0;
        return M;
    }
    // mu-weighted reference volume fraction of solid: sum_k mu_k (alpha_k rho_k) / rho_k0
    AMREX_GPU_HOST_DEVICE static Set::Scalar prefactor(const Mat &M, Set::Scalar m0, Set::Scalar m1)
    {
        return M.mu[0] * std::max(m0, 0.0) / M.rho0[0] + M.mu[1] * std::max(m1, 0.0) / M.rho0[1];
    }
    // reference volume fraction of solid (phases with mu > 0): sum_k (alpha_k rho_k) / rho_k0
    AMREX_GPU_HOST_DEVICE static Set::Scalar refvol(const Mat &M, Set::Scalar m0, Set::Scalar m1)
    {
        return ((M.mu[0] > 0.0) ? std::max(m0, 0.0) / M.rho0[0] : 0.0) + ((M.mu[1] > 0.0) ? std::max(m1, 0.0) / M.rho0[1] : 0.0);
    }
    // volume fraction occupied by solid phases, from alpha of phase 0
    AMREX_GPU_HOST_DEVICE Set::Scalar solidfrac(Set::Scalar a0) const
    {
        return ((mu[0] > 0.0) ? a0 : 0.0) + ((mu[1] > 0.0) ? 1.0 - a0 : 0.0);
    }
    // S = sum_k alpha_k dev(sigma_k) and elastic energy per volume W from the cobasis.
    // m0, m1 = (alpha rho)_k, the transported phase masses.
    AMREX_GPU_HOST_DEVICE static void stress(const Mat &M, const Set::Scalar Em[3][3], Set::Scalar m0, Set::Scalar m1,
                                             Set::Scalar S[3][3], Set::Scalar &W)
    {
        const Set::Scalar pre = prefactor(M, m0, m1);
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
    AMREX_GPU_HOST_DEVICE static Set::Scalar c2(const Mat &M, Set::Scalar a0, Set::Scalar rho)
    {
        return (4.0 / 3.0) * (M.mu[0] * a0 + M.mu[1] * (1.0 - a0)) / std::max(rho, 1.0e-300);
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
        // [2026-10-06] von Mises / Perzyna plastic relaxation of the elastic deformation (Favrie & Schmidmayer 2026)
        pp_query_default("elastic.plastic", value.plastic, 0);
        if (value.plastic)
        {
            pp_query_default("elastic.sigma_y0", value.sigma_y[0], 0.0);   // yield stress of phase 0 (<= 0: never yields)
            pp_query_default("elastic.sigma_y1", value.sigma_y[1], 0.0);   // yield stress of phase 1
            pp_query_default("elastic.Ky0", value.Ky[0], 0.0);             // plastic viscosity of phase 0 [stress x time]; 0 = perfect plasticity
            pp_query_default("elastic.Ky1", value.Ky[1], 0.0);             // plastic viscosity of phase 1
            // Compaction energy and GTN yield function of Favrie & Schmidmayer (2026); the solid must be phase 0
            pp_query_default("elastic.compaction", value.comp_on, 0);
            if (value.comp_on)
            {
                pp_query_default("elastic.chi0", value.chi[0], 0.0);       // compaction modulus of the solid [pressure]
                pp_query_default("elastic.gtn0", value.gtn[0], 0);         // 1 = GTN yield function, 0 = von Mises
                pp_query_default("elastic.comp.n", value.comp_n, 2.0);     // exponent n_k of (xi - 1)
                pp_query_default("elastic.comp.nkappa", value.comp_nk, 0.5);   // n_kappa
                pp_query_default("elastic.comp.akappa", value.comp_ak, 1.8e5); // a_kappa
                pp_query_default("elastic.gtn.q1", value.q1, 1.0);         // GTN pore-geometry parameters
                pp_query_default("elastic.gtn.q2", value.q2, 1.0);
                pp_query_default("elastic.gtn.q3", value.q3, 1.0);
            }
        }

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
              amrex::MultiFab &E_mf, amrex::MultiFab &Eold_mf, const amrex::MultiFab *sp_mf = nullptr)
    {
        if (cobasis_ic) cobasis_ic->Initialize(lev, ebasis_mf, 0.0);
        const bool have_ic = cobasis_ic != nullptr;
        for (amrex::MFIter mfi(*ebasis_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box bx = mfi.validbox();
            auto eb = ebasis_mf[lev]->array(mfi);
            auto m0 = m0_mf.const_array(mfi);
            auto m1 = m1_mf.const_array(mfi);
            amrex::Array4<const Set::Scalar> sp; if (sp_mf) sp = sp_mf->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar rv = std::max(refvol(matAt(sp, i, j, k), m0(i, j, k), m1(i, j, k)), 1.0e-12);
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
                amrex::Array4<const Set::Scalar> sp; if (sp_mf) sp = sp_mf->const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    E(i, j, k) += energy(eb, i, j, k, m0(i, j, k), m1(i, j, k), sp);  Eo(i, j, k) = E(i, j, k);
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
    void ResetFluidCells(int lev, const amrex::MultiFab &eta_mf, const amrex::MultiFab &m0_mf, const amrex::MultiFab &m1_mf, const amrex::MultiFab *sp_mf = nullptr)
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
            amrex::Array4<const Set::Scalar> sp; if (sp_mf) sp = sp_mf->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar a0 = std::min(std::max(et(i, j, k), 0.0), 1.0);
                if (solidfrac(a0) >= fcut) return;
                const Set::Scalar lam = std::cbrt(std::max(refvol(matAt(sp, i, j, k), m0(i, j, k), m1(i, j, k)), 1.0e-12));
                for (int n = 0; n < NEB; ++n) eb(i, j, k, n) = 0.0;
                for (int b = 0; b < 3; ++b) eb(i, j, k, comp(b, b)) = lam;
            });
        }
    }

    // PLASTIC RELAXATION (elastic.plastic), operator split, once per step after the flow update.
    // In every cell that holds solid: Kirchhoff-type deviator of the solid  tau = -mu dev(Gt^2 - Gt)  (the mixture S
    // divided by the reference solid volume fraction: independent of how much solid the cell holds), von Mises stress
    // sigma_eq = sqrt(3/2 tau:tau).  If sigma_eq > sigma_y the cobasis is rotated-stretched back toward the yield
    // surface along the flow direction N = 3 tau/(2 sigma_eq) (traceless: volume preserving, det E is kept exactly):
    //     E <- E exp(dgamma N),      dgamma = dt <sigma_eq(dgamma) - sigma_y>/K_y     (backward Euler: the radial return
    // of rate-independent plasticity when K_y -> 0).  dgamma is found by a secant iteration on the true non-linear
    // sigma_eq(dgamma), started from the small-strain value (sigma_eq - sigma_y)/(3 mu + K_y/dt).
    // Material constants in a cell with two solids are weighted with their reference volume fractions.
    // ENERGY: the elastic energy W removed here stays in the conserved rho E and reappears as internal energy of the
    // phases at the next re-initialisation (rho E - kinetic - W): plastic work heats the solid, nothing is lost.
    void PlasticRelax(int lev, Set::Scalar dt, const amrex::MultiFab &eta_mf, const amrex::MultiFab &m0_mf, const amrex::MultiFab &m1_mf, const amrex::MultiFab *sp_mf = nullptr, amrex::MultiFab *cf_mf = nullptr)
    {
        if (!plastic) return;
        for (amrex::MFIter mfi(*ebasis_mf[lev], false); mfi.isValid(); ++mfi)
        {
            auto eb = ebasis_mf[lev]->array(mfi); auto pd = plastic_mf[lev]->array(mfi);
            auto et = eta_mf.const_array(mfi); auto m0 = m0_mf.const_array(mfi); auto m1 = m1_mf.const_array(mfi);
            amrex::Array4<const Set::Scalar> sp; if (sp_mf) sp = sp_mf->const_array(mfi);
            amrex::Array4<Set::Scalar> cf; if (cf_mf && comp_on) cf = cf_mf->array(mfi);
            const bool do_comp = comp_on && cf_mf;
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                pd(i, j, k, 0) = 0.0; pd(i, j, k, 1) = 0.0;
                const Mat M = matAt(sp, i, j, k);
                const Set::Scalar v0 = (M.mu[0] > 0.0) ? std::max(m0(i, j, k), 0.0) / M.rho0[0] : 0.0;      // reference volume fractions
                const Set::Scalar v1 = (M.mu[1] > 0.0) ? std::max(m1(i, j, k), 0.0) / M.rho0[1] : 0.0;
                const Set::Scalar vs = v0 + v1;
                if (!(vs > 1.0e-12)) return;
                const Set::Scalar a0 = std::min(std::max(et(i, j, k), 0.0), 1.0);
                if (fluid_cut > 0.0 && solidfrac(a0) < fluid_cut) return;                                // fluid cell: reset elsewhere
                const Set::Scalar w0 = v0 / vs, w1 = v1 / vs;
                const Set::Scalar mus = w0 * M.mu[0] + w1 * M.mu[1];
                // a phase that never yields (sigma_y <= 0) makes the mixture cell rigid-plastic-free only if it is alone
                const Set::Scalar sy = w0 * ((M.sigma_y[0] > 0.0) ? M.sigma_y[0] : 1.0e300) + w1 * ((M.sigma_y[1] > 0.0) ? M.sigma_y[1] : 1.0e300);
                const Set::Scalar ky = w0 * M.Ky[0] + w1 * M.Ky[1];
                if (!(sy < 1.0e299)) return;
                Set::Scalar Em[3][3], S[3][3], W;
                load(eb, i, j, k, Em);
                const Set::Scalar pre = prefactor(M, m0(i, j, k), m1(i, j, k));
                if (!(pre > 0.0)) return;
                // mixture S = -pre dev(Gt^2 - Gt), pre = sum_k mu_k v_k.  The solid's own Kirchhoff-type deviator is
                // tau = -mus dev(Gt^2 - Gt) = S / vs  (vs = reference solid volume fraction = alpha_s rho_s / rho_s0):
                // it does not depend on how much solid the cell holds, and in a pure solid it is (rho_0/rho) x Cauchy.
                const Set::Scalar scale = 1.0 / vs;
                auto seq = [&](const Set::Scalar E_[3][3], Set::Scalar T_[3][3]) {
                    Set::Scalar S_[3][3], W_; stress(M, E_, m0(i, j, k), m1(i, j, k), S_, W_);
                    Set::Scalar q = 0.0;
                    for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) { T_[a][c] = scale * S_[a][c]; q += T_[a][c] * T_[a][c]; }
                    return std::sqrt(1.5 * q);
                };
                Set::Scalar tau[3][3];
                const Set::Scalar s0 = seq(Em, tau);
                pd(i, j, k, 1) = s0;
                if (do_comp && M.gtn[0] > 0.5 && v0 > 0.0)
                {
                    // GTN yield function with Perzyna viscosity (see struct Comp).  One unknown, dgamma >= 0:
                    //   E <- E exp(dgamma 3 tau/sigma_y),  alpha_ref <- alpha_ref exp(dgamma dg/dpi),
                    //   K_y dgamma/dt = g(E, alpha_ref)   (backward Euler; g decreases along the return)
                    const Set::Scalar as = std::min(std::max(a0, 1.0e-6), 1.0), por = 1.0 - as, rsr = v0 / as;    // porosity, rho_s/rho_r
                    const Set::Scalar ar0 = cf(i, j, k), cq = 1.5 * q2 * rsr / sy;
                    auto gval = [&](Set::Scalar se, Set::Scalar ar, Set::Scalar *dgdpi) {
                        const Set::Scalar x = std::min(std::max(cq * compaction(M, as, m0(i, j, k), ar).pis, -50.0), 50.0);
                        if (dgdpi) *dgdpi = 2.0 * q1 * por * 1.5 * q2 * rsr * std::sinh(x);
                        return sy * ((se / sy) * (se / sy) + 2.0 * q1 * por * std::cosh(x) - 1.0 - q3 * por * por);
                    };
                    Set::Scalar dgp;
                    const Set::Scalar g0v = gval(s0, ar0, &dgp);
                    if (!(g0v > 0.0)) return;
                    Set::Scalar Ng[3][3];
                    for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) Ng[a][c] = 3.0 * tau[a][c] / sy;
                    const Set::Scalar detg = Em[0][0] * (Em[1][1] * Em[2][2] - Em[1][2] * Em[2][1]) - Em[0][1] * (Em[1][0] * Em[2][2] - Em[1][2] * Em[2][0])
                                           + Em[0][2] * (Em[1][0] * Em[2][1] - Em[1][1] * Em[2][0]);
                    auto applyg = [&](Set::Scalar g, Set::Scalar En[3][3]) {
                        Set::Scalar P[3][3], T[3][3], A[3][3];
                        for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) { A[a][c] = g * Ng[a][c]; P[a][c] = ((a == c) ? 1.0 : 0.0) + A[a][c]; T[a][c] = A[a][c]; }
                        for (int n = 2; n <= 10; ++n)
                        {
                            Set::Scalar T2[3][3];
                            for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) T2[a][c] = (T[a][0] * A[0][c] + T[a][1] * A[1][c] + T[a][2] * A[2][c]) / n;
                            for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) { T[a][c] = T2[a][c]; P[a][c] += T2[a][c]; }
                        }
                        for (int b = 0; b < 3; ++b) for (int c = 0; c < 3; ++c) En[b][c] = Em[b][0] * P[0][c] + Em[b][1] * P[1][c] + Em[b][2] * P[2][c];
                        const Set::Scalar det = En[0][0] * (En[1][1] * En[2][2] - En[1][2] * En[2][1]) - En[0][1] * (En[1][0] * En[2][2] - En[1][2] * En[2][0])
                                              + En[0][2] * (En[1][0] * En[2][1] - En[1][1] * En[2][0]);
                        const Set::Scalar f = (det * detg > 0.0) ? std::cbrt(detg / det) : 1.0;
                        for (int b = 0; b < 3; ++b) for (int c = 0; c < 3; ++c) En[b][c] *= f;
                    };
                    const Set::Scalar kd = ky / dt;
                    Set::Scalar En[3][3], T_[3][3];
                    auto resg = [&](Set::Scalar g) {       // kd g - g_GTN(g): increasing in g, negative at g = 0
                        applyg(g, En);
                        const Set::Scalar ar = ar0 * std::exp(std::min(std::max(g * dgp, -1.0), 1.0));
                        return kd * g - gval(seq(En, T_), ar, nullptr);
                    };
                    // bracket [0, gmax]: gmax = sigma_y/(6 mu) is where the deviator has relaxed to zero (linearised:
                    // tau <- tau (1 - 6 mu dgamma/sigma_y)); beyond it the fixed direction would REVERSE the deviator and
                    // the residual is no longer monotone.  With viscosity the explicit value g/kd is a tighter bound.
                    Set::Scalar ga = 0.0, gb = sy / (6.0 * std::max(mus, 1.0e-300));
                    if (kd > 0.0) gb = std::min(gb, g0v / kd);
                    Set::Scalar rb = resg(gb);
                    if (rb > 0.0)
                        for (int it = 0; it < 60; ++it)
                        {
                            const Set::Scalar gm = 0.5 * (ga + gb), rm = resg(gm);
                            if (rm > 0.0) gb = gm; else ga = gm;
                            if (gb - ga <= 1.0e-12 * gb) break;
                        }
                    const Set::Scalar gsol = (rb > 0.0) ? 0.5 * (ga + gb) : gb;
                    applyg(gsol, En);
#if AMREX_SPACEDIM == 2
                    eb(i, j, k, 0) = En[0][0]; eb(i, j, k, 1) = En[0][1]; eb(i, j, k, 2) = En[1][0]; eb(i, j, k, 3) = En[1][1]; eb(i, j, k, 4) = En[2][2];
#else
                    for (int b = 0; b < 3; ++b) for (int c = 0; c < 3; ++c) eb(i, j, k, 3 * b + c) = En[b][c];
#endif
                    cf(i, j, k) = std::min(ar0 * std::exp(std::min(std::max(gsol * dgp, -1.0), 1.0)), 1.0);
                    pd(i, j, k, 0) = gsol / dt;
                    pd(i, j, k, 1) = seq(En, T_);
                    return;
                }
                if (!(s0 > sy)) return;
                Set::Scalar N[3][3];
                for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) N[a][c] = 1.5 * tau[a][c] / s0;
                // E exp(g N), exp by its series (g N is a strain increment, << 1), then det restored exactly
                const Set::Scalar det0 = Em[0][0] * (Em[1][1] * Em[2][2] - Em[1][2] * Em[2][1]) - Em[0][1] * (Em[1][0] * Em[2][2] - Em[1][2] * Em[2][0])
                                       + Em[0][2] * (Em[1][0] * Em[2][1] - Em[1][1] * Em[2][0]);
                auto apply = [&](Set::Scalar g, Set::Scalar En[3][3]) {
                    Set::Scalar P[3][3], T[3][3], A[3][3];
                    for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) { A[a][c] = g * N[a][c]; P[a][c] = ((a == c) ? 1.0 : 0.0) + A[a][c]; T[a][c] = A[a][c]; }
                    for (int n = 2; n <= 8; ++n)
                    {
                        Set::Scalar T2[3][3];
                        for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) T2[a][c] = (T[a][0] * A[0][c] + T[a][1] * A[1][c] + T[a][2] * A[2][c]) / n;
                        for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) { T[a][c] = T2[a][c]; P[a][c] += T2[a][c]; }
                    }
                    for (int b = 0; b < 3; ++b) for (int c = 0; c < 3; ++c) En[b][c] = Em[b][0] * P[0][c] + Em[b][1] * P[1][c] + Em[b][2] * P[2][c];
                    const Set::Scalar det = En[0][0] * (En[1][1] * En[2][2] - En[1][2] * En[2][1]) - En[0][1] * (En[1][0] * En[2][2] - En[1][2] * En[2][0])
                                          + En[0][2] * (En[1][0] * En[2][1] - En[1][1] * En[2][0]);
                    const Set::Scalar f = (det * det0 > 0.0) ? std::cbrt(det0 / det) : 1.0;
                    for (int b = 0; b < 3; ++b) for (int c = 0; c < 3; ++c) En[b][c] *= f;
                };
                // residual r(g) = g - dt <sigma_eq(g) - sigma_y>/K_y   (K_y = 0: r = sigma_eq(g) - sigma_y, scaled)
                const Set::Scalar kd = ky / dt;
                auto res = [&](Set::Scalar g, Set::Scalar En[3][3]) {
                    Set::Scalar T_[3][3]; apply(g, En); const Set::Scalar s = seq(En, T_);
                    return (3.0 * mus + kd) * g - (s - sy) - 3.0 * mus * g;        // = kd g - (sigma_eq(g) - sy)
                };
                Set::Scalar En[3][3];
                Set::Scalar g1 = (s0 - sy) / (3.0 * mus + kd), r1 = res(g1, En);
                Set::Scalar g0 = 0.0, r0 = -(s0 - sy);
                for (int it = 0; it < 12 && std::abs(r1) > 1.0e-10 * sy; ++it)
                {
                    const Set::Scalar d = (r1 - r0);
                    if (!(std::abs(d) > 0.0)) break;
                    const Set::Scalar g2 = std::max(g1 - r1 * (g1 - g0) / d, 0.0);
                    g0 = g1; r0 = r1; g1 = g2; r1 = res(g1, En);
                }
                apply(g1, En);
#if AMREX_SPACEDIM == 2
                eb(i, j, k, 0) = En[0][0]; eb(i, j, k, 1) = En[0][1]; eb(i, j, k, 2) = En[1][0]; eb(i, j, k, 3) = En[1][1]; eb(i, j, k, 4) = En[2][2];
#else
                for (int b = 0; b < 3; ++b) for (int c = 0; c < 3; ++c) eb(i, j, k, 3 * b + c) = En[b][c];
#endif
                pd(i, j, k, 0) = g1 / dt;
                Set::Scalar T_[3][3]; pd(i, j, k, 1) = seq(En, T_);
            });
        }
    }

    // ---------------------------------------------------------------------------------------
    // Per-cell / per-face kernels, called inside Hydro2's loops.
    // ---------------------------------------------------------------------------------------
    // elastic energy per volume of one cell
    template <class A, class SP>
    AMREX_GPU_HOST_DEVICE Set::Scalar energy(const A &eb, int i, int j, int k, Set::Scalar m0, Set::Scalar m1, const SP &sp) const
    {
        Set::Scalar Em[3][3], S[3][3], W;
        load(eb, i, j, k, Em);
        stress(matAt(sp, i, j, k), Em, m0, m1, S, W);
        return W;
    }
    // ... plus the compaction energy (as = solid volume fraction, aref = cfun); equal to the above with compaction off
    template <class A, class SP>
    AMREX_GPU_HOST_DEVICE Set::Scalar energy(const A &eb, int i, int j, int k, Set::Scalar m0, Set::Scalar m1, const SP &sp,
                                             Set::Scalar as, Set::Scalar aref) const
    {
        const Set::Scalar W = energy(eb, i, j, k, m0, m1, sp);
        return comp_on ? W + compaction(matAt(sp, i, j, k), as, m0, aref).W : W;
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
    template <class A, class B, class C, class SP, class CF>
    AMREX_GPU_HOST_DEVICE void faceState(Solver::Local::FluidRiemann::State &st, const A &eb, const B &eta,
                                         const C &m0, const C &m1, int ci, int cj, int ck, int dir, int sd, const SP &sp, const CF &cf) const
    {
        Set::Scalar Em[3][3], S[3][3], W, Ef[NEB];
        faceCobasis(eb, eta, ci, cj, ck, dir, sd, Ef);
        emFrom(Ef, Em);
        const Mat M = matAt(sp, ci, cj, ck);
        stress(M, Em, m0(ci, cj, ck), m1(ci, cj, ck), S, W);
        const int t1 = (dir + 1) % AMREX_SPACEDIM;
        st.S_nn = S[dir][dir];
        st.S_nt = S[dir][t1];
#if AMREX_SPACEDIM == 3
        st.S_nt2 = S[dir][(dir + 2) % AMREX_SPACEDIM];
#endif
        const Set::Scalar a0 = std::min(std::max(eta(ci, cj, ck), 0.0), 1.0);
        if (comp_on) st.S_nn -= a0 * compaction(M, a0, m0(ci, cj, ck), cf(ci, cj, ck)).Pc;      // true -alpha_s p_s (see struct Comp)
        st.c2_el = c2(M, a0, m0(ci, cj, ck) + m1(ci, cj, ck));
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
    template <class A, class B, class C, class R, class FluxT, class SP>
    AMREX_GPU_HOST_DEVICE void cobasisRHS(const R &rhs, const R &diag, const A &eb, const B &eta, const C &m0, const C &m1,
                                          const FluxT *const Fl[AMREX_SPACEDIM], const FluxT *const Fh[AMREX_SPACEDIM],
                                          const Set::Scalar *DX, int i, int j, int k, const SP &sp) const
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
        stress(matAt(sp, i, j, k), Em, m0(i, j, k), m1(i, j, k), S, W);
        diag(i, j, k, 0) = S[0][0]; diag(i, j, k, 1) = S[1][1]; diag(i, j, k, 2) = S[2][2];
        diag(i, j, k, 3) = S[0][1]; diag(i, j, k, 4) = S[0][2]; diag(i, j, k, 5) = S[1][2];
        diag(i, j, k, 6) = W;
    }
};
}
#endif
