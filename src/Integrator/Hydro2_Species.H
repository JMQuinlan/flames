#ifndef INTEGRATOR_HYDRO2_SPECIES_H
#define INTEGRATOR_HYDRO2_SPECIES_H
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include "Set/Base.H"
#include "BC/BC.H"
#include "IC/IC.H"
#include "Solver/EOS/Tammann.H"
#include "Integrator/Hydro2_Pele.H"
namespace Integrator {
// Hydro2_Species.H -- SPECIES inside the two phases of Hydro2, default OFF (species.on = 0).
// Design: docs/design/Hydro2_MultiSpecies_Chemistry_Design.md.  THIS FILE IS STAGE 1: PASSIVE TRANSPORT ONLY.
// The equation of state, the relaxation and the flow do not see the species yet (the flow is bit-identical with
// species.on = 1); chemistry, species thermodynamics and phase change on species are later stages.
//
// MODEL.  The solver keeps its two phases (volume fraction eta, phase partial densities rho_eta0 = alpha_0 rho_0 and
// rho_eta1 = alpha_1 rho_1).  Each phase is a MIXTURE of species; a substance present as a liquid and as a vapour is
// two entries, one in each phase.  The transported field ("species", s = N0 + N1 components) is the species partial
// density  m_i = alpha_k rho_k Y_{k,i}   (k = phase of species i),   sum_{i in k} m_i = alpha_k rho_k.
//     d m_i/dt + div(F_k Y_i) = 0,   F_k = mass flux of phase k from the Riemann solver, Y_i upwinded on its sign.
//
// CLOSURE (rule set by the user, 2026-10-06): the small fractions are advected first and the LARGEST species of the
// phase takes the remainder, so trace species are never lost to a normalisation:
//   * face:  fluxes of all species but the majority one (largest Y in the upwind cell) are F_k Y_i, accumulated from
//            the smallest up; the majority species gets  F_k - sum(others).  The species fluxes then add up to the
//            phase mass flux EXACTLY, so sum_i m_i = alpha_k rho_k is kept to round-off by the flux update itself.
//   * cell:  after each Runge-Kutta stage negatives (round-off) are clipped and the majority species of the cell is
//            reset to  alpha_k rho_k - sum(others);  only if the others alone exceed the phase mass (never seen in the
//            tests) they are scaled down together.
//
// INPUT
//   species.on = 1
//   species.phase0.names = A B ...        species of phase 0 (the eta = 1 phase), may be empty
//   species.phase1.names = C D ...        species of phase 1
//   species.ic.type = expression | constant      mass fractions Y WITHIN THEIR PHASE, one component per species in the
//   species.ic.expression.regionN = "..."        order phase0 names then phase1 names.  Give EVERY species (the
//                                                largest given value of a cell is the one closed to the phase mass,
//                                                which only removes the round-off of a set that sums to 1)
//   species.bc.type.xlo = ...             boundary condition of the partial densities (BC::Expression, s components);
//                                         default zero gradient -- REQUIRED explicitly (periodic) on periodic domains
// OUTPUT  plot fields species_<name> (partial density) and Y_<name> (mass fraction within the phase).
// LIMITS  single level (no reflux of the species fluxes yet); first-order (donor-cell) face values, like the alpha row.
// STAGE 2a (species.eos = mixture): species-dependent gamma and c_v inside a phase -- see the members below.
class Species {
public:
    static constexpr int NMAX = 24;                  // stack-array bound per phase in the kernels
    int on = 0;
    int n0 = 0, n1 = 0;                              // species in phase 0 / phase 1
    std::vector<std::string> names;                  // phase 0 first, then phase 1
    Set::Field<Set::Scalar> mf, old_mf;              // species partial densities (evolved)
    Set::Field<Set::Scalar> Y_mf;                    // mass fractions within the phase (diagnostic, also the IC carrier)
    BC::BC<Set::Scalar> *bc = nullptr;
    IC::IC<Set::Scalar> *ic = nullptr;
    int ns() const { return n0 + n1; }

    // STAGE 2a (species.eos = mixture, default fixed = stage 1, the flow does not see the species).
    // Every species i of a phase is a stiffened gas with its own gamma_i and c_v,i and the STIFFNESS pi OF ITS PHASE
    // (eosK.p0); the species of a phase share pressure and temperature.  The phase is then EXACTLY a stiffened gas
    // with      c_v = sum_i Y_i c_v,i,     gamma = sum_i Y_i gamma_i c_v,i / c_v,     pi = pi_k
    // (for pi = 0 this is the ideal mixture of calorically perfect gases; molar masses enter through
    // R_i = (gamma_i - 1) c_v,i).  The solver reads these per cell wherever it used the constants of eosK.
    // Different pi inside one phase (two solids in the solid phase) is NOT covered: that mixture is not a
    // stiffened gas and needs the local fit of the design document (section 3.4) -- stage 2b.
    //   species.eos = mixture
    //   species.phase0.gamma = ...   species.phase0.cv = ...     one value per species of phase 0
    //   species.phase1.gamma = ...   species.phase1.cv = ...
    int eos_mix = 0;
    // species.order = 1 (default): mass fraction of the upwind cell at a face.  2: upwind value + half a van Leer
    // (harmonic) limited slope, i.e. second order in smooth regions and still bounded by the neighbouring cells.
    // First order in the LAB frame smears radicals forward through a captured detonation front (e-folding length
    // ~ one cell at post-shock speeds) and removes the induction zone of chain-branching chemistry.
    int order = 1;
    // species.order = 3: THINC face value in the mixed, monotone cells (see Parse); beta_0 and the mixed-cell threshold
    Set::Scalar thinc_beta = 2.5, thinc_eps = 1.0e-4;
    Set::Scalar sgam[2 * NMAX] = {}, scv[2 * NMAX] = {};
    // STAGE 2b: species of one phase with DIFFERENT stiffness (two solids in the solid phase).  Given
    //   species.phaseK.pinf = ...   species.phaseK.rho_ref = ...      (one value per species; both or neither)
    // the phase constants of a cell follow the volume-weighted (isobaric) closure of the five-equation models,
    //   z_i = (Y_i / rho_ref,i) / sum_j (Y_j / rho_ref,j)                     volume fraction of species i in its phase
    //   1/(gamma - 1) = sum z_i/(gamma_i - 1),   gamma pi/(gamma - 1) = sum z_i gamma_i pi_i/(gamma_i - 1),
    //   c_v = sum Y_i c_v,i
    // which is exact where a single species fills the phase and a CLOSURE in the numerically mixed cells of a
    // solid | solid contact (the volume fractions are taken at the reference densities, not from a p-T equilibrium).
    int has_pi[2] = {0, 0};
    Set::Scalar spi[2 * NMAX] = {}, srho[2 * NMAX] = {};
    struct Eos                                        // copyable, captured by value in the kernels
    {
        int on = 0, n0 = 0, n1 = 0;
        Set::Scalar g[2 * NMAX], cv[2 * NMAX];
        int has_pi[2] = {0, 0};
        Set::Scalar pi[2 * NMAX], rr[2 * NMAX];       // stage 2b: stiffness and reference density per species
        Pele::Thermo th;                              // species.eos = pele: thermally perfect species (Hydro2_Pele.H)
        // gam, cvm come in as the phase defaults.  es = specific sensible energy E_k / m_k of the phase in this cell;
        // only the pele equation of state uses it (es <= 0: not known, the low-temperature limit is returned).
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void Mix(const amrex::Array4<const Set::Scalar> &m, int i, int j, int k, int ph,
                 Set::Scalar &gam, Set::Scalar &cvm, const Set::Scalar es = -1.0, Set::Scalar *pinf = nullptr,
                 const Set::Scalar rhok = -1.0) const      // rhok = phase density (pele, frozen sound speed only)
        {
            const int n = ph ? n1 : n0, c0 = ph ? n0 : 0;
            if (!on || n == 0) return;
            if (has_pi[ph])     // stage 2b: volume-weighted closure, also returns the stiffness of the cell
            {
                Set::Scalar sm = 0.0, sc = 0.0, sv = 0.0, a = 0.0, b = 0.0;
                for (int s = 0; s < n; ++s)
                {
                    const Set::Scalar mi = std::max(m(i, j, k, c0 + s), 0.0), vi = mi / rr[c0 + s];
                    sm += mi; sc += mi * cv[c0 + s]; sv += vi;
                    a += vi / (g[c0 + s] - 1.0); b += vi * g[c0 + s] * pi[c0 + s] / (g[c0 + s] - 1.0);
                }
                if (!(sm > 1.0e-300) || !(sv > 0.0)) return;
                cvm = sc / sm; gam = 1.0 + sv / a;
                if (pinf) *pinf = (b / sv) * (gam - 1.0) / gam;
                return;
            }
            if (th.on && ph == th.ph)
            {
                Set::Scalar Y[Pele::NS], sm = 0.0;
                for (int s = 0; s < Pele::NS; ++s) { Y[s] = std::max(m(i, j, k, c0 + s), 0.0); sm += Y[s]; }
                if (!(sm > 1.0e-300)) return;
                for (int s = 0; s < Pele::NS; ++s) Y[s] /= sm;
                if (th.frozen_c && pinf && rhok > 0.0 && es > 0.0) th.GammaPiCv(es, rhok, Y, gam, *pinf, cvm);
                else th.GammaCv(es, Y, gam, cvm);
                return;
            }
            Set::Scalar sm = 0.0, sc = 0.0, sp = 0.0;
            for (int s = 0; s < n; ++s)
            {
                const Set::Scalar mi = std::max(m(i, j, k, c0 + s), 0.0);
                sm += mi; sc += mi * cv[c0 + s]; sp += mi * cv[c0 + s] * g[c0 + s];
            }
            if (!(sm > 1.0e-300) || !(sc > 0.0)) return;
            cvm = sc / sm; gam = sp / sc;
        }
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        Solver::EOS::Tammann Phase(const amrex::Array4<const Set::Scalar> &m, int i, int j, int k, int ph,
                                   const Solver::EOS::Tammann &base, const Set::Scalar es = -1.0, const Set::Scalar rhok = -1.0) const
        {
            Set::Scalar gam = base.Gamma(), cvm = base.Cv(), pin = base.P0();
            Mix(m, i, j, k, ph, gam, cvm, es, &pin, rhok);
            return Solver::EOS::Tammann(gam, pin, cvm, gam * cvm);
        }
        // start-up (pele): specific sensible energy of the phase from its pressure and density; -1 otherwise
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        Set::Scalar EsFromPRho(const amrex::Array4<const Set::Scalar> &m, int i, int j, int k, int ph,
                               const Set::Scalar p, const Set::Scalar rho) const
        {
            if (!on || !th.on || ph != th.ph) return -1.0;
            const int c0 = ph ? n0 : 0;
            Set::Scalar Y[Pele::NS], sm = 0.0;
            for (int s = 0; s < Pele::NS; ++s) { Y[s] = std::max(m(i, j, k, c0 + s), 0.0); sm += Y[s]; }
            if (!(sm > 1.0e-300)) return -1.0;
            for (int s = 0; s < Pele::NS; ++s) Y[s] /= sm;
            return th.EsFromPRho(p, rho, Y);
        }
    };
    Pele::Thermo pele;                               // filled at parse when species.eos = pele
    Eos eos() const
    {
        Eos e; e.on = (on && eos_mix) ? 1 : 0; e.n0 = n0; e.n1 = n1; e.th = pele;
        e.has_pi[0] = has_pi[0]; e.has_pi[1] = has_pi[1];
        for (int s = 0; s < 2 * NMAX; ++s) { e.g[s] = sgam[s]; e.cv[s] = scv[s]; e.pi[s] = spi[s]; e.rr[s] = (srho[s] > 0.0) ? srho[s] : 1.0; }
        return e;
    }

    // Face fluxes of the n species of ONE phase.  F = phase mass flux through the face, Y = mass fractions in the
    // upwind cell.  Smallest first, the largest takes the remainder (see the header).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void FaceFluxes(const Set::Scalar F, const Set::Scalar *Y, const int n, Set::Scalar *out)
    {
        if (n <= 0) return;
        int imax = 0;
        for (int i = 1; i < n; ++i) if (Y[i] > Y[imax]) imax = i;
        // accumulate in ascending order of Y (insertion order on a small array): the sum of the small terms is exact
        // to the last bit that matters before it is taken off the phase flux
        int idx[NMAX]; int m = 0;
        for (int i = 0; i < n; ++i) if (i != imax) idx[m++] = i;
        for (int a = 1; a < m; ++a) { const int v = idx[a]; int b = a - 1; while (b >= 0 && Y[idx[b]] > Y[v]) { idx[b + 1] = idx[b]; --b; } idx[b + 1] = v; }
        Set::Scalar sum = 0.0;
        for (int a = 0; a < m; ++a) { out[idx[a]] = F * Y[idx[a]]; sum += out[idx[a]]; }
        out[imax] = F - sum;
    }

    // Cell closure of the n species of one phase to the phase partial density mk.  Returns what had to be clipped
    // (sum of the negative parts removed), a diagnostic that should stay at round-off.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static Set::Scalar CloseCell(Set::Scalar *m, const int n, const Set::Scalar mk)
    {
        if (n <= 0) return 0.0;
        Set::Scalar clipped = 0.0;
        for (int i = 0; i < n; ++i) if (m[i] < 0.0) { clipped -= m[i]; m[i] = 0.0; }
        int imax = 0;
        for (int i = 1; i < n; ++i) if (m[i] > m[imax]) imax = i;
        Set::Scalar others = 0.0;
        for (int i = 0; i < n; ++i) if (i != imax) others += m[i];
        const Set::Scalar mkp = std::max(mk, 0.0);
        if (others <= mkp) m[imax] = mkp - others;
        else { const Set::Scalar f = mkp / others; for (int i = 0; i < n; ++i) m[i] = (i == imax) ? 0.0 : m[i] * f; }
        return clipped;
    }
};
}
#endif
