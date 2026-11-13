#ifndef INTEGRATOR_HYDRO2_ENERGETIC_H
#define INTEGRATOR_HYDRO2_ENERGETIC_H
#include <algorithm>
#include <cmath>
#include <string>
#include "Set/Base.H"
namespace Integrator {
// Hydro2_Energetic.H -- ENERGETIC (condensed-phase explosive) MATERIAL, default OFF (energetic.on = 0).
// Needs species.on = 1.  2026-10-09.
//
// MODEL.  The six-equation two-phase form of the reactive mixture of Petitpas, Saurel, Franquet & Chinnayya,
// Shock Waves 19 (2009) 377, used by Schoch, Nordin-Bates & Nikiforakis, JCP 252 (2013) 163 and by Wallis, Barton &
// Nikiforakis, Comput. Struct. 254 (2021) 106578 (their "idealised explosive", Table 2): the REACTANT is a species
// of the condensed phase 0 (it may share that phase with inert solids, e.g. a copper confiner given as another
// species with elastic.species.mu), the PRODUCTS are a species of the gas phase 1.  Reaction is a mass transfer
// from phase 0 to phase 1 at a rate given by the reaction-progress law; the two phases keep their own equations of
// state and are brought to one pressure by the relaxation step that follows (not to one temperature: Wallis et al.
// close their one-material mixture with p and T equilibrium instead).
//
//   lambda = m_r / (m_r + m_p)   mass fraction of unreacted explosive of the cell (1 = reactant, 0 = products);
//                                m_r, m_p = partial densities of the reactant and product species
//   law = sqrt :   d lambda/dt = -k sqrt(lambda)   while  p > p_ign     (Wallis eq. 61; exact step
//                                sqrt(lambda) -> max(sqrt(lambda) - k dt/2, 0))
//   law = pressure : d lambda/dt = -k lambda (p/p_ref)^n  while p > p_ign   (pressure-dependent burn, exact step)
//
// STEP (operator split, once per step after the flow update, before the relaxation).  For the mass dm that reacts:
//   phase masses and species:  (alpha rho)_0 -= dm,  (alpha rho)_1 += dm,  m_r -= dm,  m_p += dm
//   volume:    the reacted mass takes the volume it had in phase 0 with it,  alpha_0 -= alpha_0 dm/(alpha rho)_0
//              (the products start at the density of the reactant; the relaxation then expands them)
//   energy:    it takes its specific energy e_0 = E_0/(alpha rho)_0 with it and receives the heat of reaction,
//              E_0 -= e_0 dm,   E_1 += (e_0 + Q) dm,   rho E += Q dm     (sensible-energy convention of Hydro2_Chem.H)
// Momentum and mixture density do not change.
//
// INPUT   energetic.on = 1
//         energetic.reactant = <species of phase 0>     energetic.product = <species of phase 1>
//         energetic.law = sqrt | pressure               energetic.k (1/s)      energetic.Q (energy per mass)
//         energetic.p_ign (pressure above which the reaction runs)
//         energetic.p_ref, energetic.n                  (law = pressure only)
class Energetic {
public:
    int on = 0;
    std::string reactant, product;
    int ir = -1, ip = -1;
    int law = 0;                             // 0 = sqrt, 1 = pressure
    Set::Scalar k = 0.0, Q = 0.0, p_ign = 0.0, p_ref = 1.0, n = 1.0;

    // new unreacted fraction after dt
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static Set::Scalar Step(const int law, const Set::Scalar lam, const Set::Scalar p, const Set::Scalar k,
                            const Set::Scalar p_ref, const Set::Scalar n, const Set::Scalar dt)
    {
        if (law == 0) { const Set::Scalar s = std::sqrt(lam) - 0.5 * k * dt; return (s > 0.0) ? s * s : 0.0; }
        return lam * std::exp(-k * std::pow(std::max(p, 0.0) / p_ref, n) * dt);
    }
};
}
#endif
