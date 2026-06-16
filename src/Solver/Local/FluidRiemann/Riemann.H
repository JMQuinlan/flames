// 6-equation diffuse-interface Riemann state and flux.
//
// State and Flux carry the conservative variables of the six-equation
// (two-pressure, two-energy) model:
//     q = ( alpha, (alpha rho)_0, (alpha rho)_1,
//           rho u, rho v,
//           (alpha rho e)_0, (alpha rho e)_1,
//           rho E_redundant )
//
// Reference (state vector / flux): Schmidmayer-Bryngelson-Colonius JCP 402 (2020)
// eq. (13), plus the redundant total-energy row from eq. (16).  HLLC star-state
// algebra follows Saurel-Petitpas-Berry JCP 228 (2009) Section 3.1.2.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_H
#define SOLVER_LOCAL_FLUIDRIEMANN_H

#include "Set/Base.H"
#include "Set/Set.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// Riemann-problem state for the 6-equation model.
///
/// All members are the CONSERVATIVE primary variables (the per-phase
/// internal energies and the redundant total energy are PRIMARIES in
/// the 6-eq formulation - Saurel 2009 eq. (II.6); Schmidmayer 2020
/// eq. (13)+(16)).
struct State
{
    // Volume fraction
    Set::Scalar alpha       = 0.0;    // alpha_1

    // Phase masses (canonical)
    Set::Scalar alpha_rho_0 = 0.0;    // (alpha_1 rho_1)
    Set::Scalar alpha_rho_1 = 0.0;    // (alpha_2 rho_2)

    // Mixture momentum
    Set::Scalar M_normal    = 0.0;
    Set::Scalar M_tangent   = 0.0;

    // Per-phase internal energies (canonical)
    Set::Scalar E0          = 0.0;    // (alpha_1 rho_1 e_1)
    Set::Scalar E1          = 0.0;    // (alpha_2 rho_2 e_2)

    // Redundant mixture total energy (for shock fix-up via reinit)
    Set::Scalar E_total     = 0.0;    // rho E

    // Per-phase Tammann/stiffened-gas constants (not advected; copied per call).
    Set::Scalar gamma0      = NAN;
    Set::Scalar pi0         = NAN;
    Set::Scalar gamma1      = NAN;
    Set::Scalar pi1         = NAN;

    State() = default;

    State(Set::Scalar a_alpha,
          Set::Scalar a_alpha_rho_0,
          Set::Scalar a_alpha_rho_1,
          Set::Scalar a_M_normal,
          Set::Scalar a_M_tangent,
          Set::Scalar a_E0,
          Set::Scalar a_E1,
          Set::Scalar a_E_total,
          Set::Scalar a_gamma0 = NAN,
          Set::Scalar a_pi0    = NAN,
          Set::Scalar a_gamma1 = NAN,
          Set::Scalar a_pi1    = NAN)
        : alpha(a_alpha),
          alpha_rho_0(a_alpha_rho_0),
          alpha_rho_1(a_alpha_rho_1),
          M_normal(a_M_normal),
          M_tangent(a_M_tangent),
          E0(a_E0),
          E1(a_E1),
          E_total(a_E_total),
          gamma0(a_gamma0),
          pi0(a_pi0),
          gamma1(a_gamma1),
          pi1(a_pi1)
    {
    }

    friend std::ostream &
    operator<<(std::ostream &os, const State &s)
    {
        os << "alpha=" << s.alpha
           << ", arh0=" << s.alpha_rho_0
           << ", arh1=" << s.alpha_rho_1
           << ", Mn="   << s.M_normal
           << ", Mt="   << s.M_tangent
           << ", E0="   << s.E0
           << ", E1="   << s.E1
           << ", E="    << s.E_total
           << ", g0="   << s.gamma0
           << ", pi0="  << s.pi0
           << ", g1="   << s.gamma1
           << ", pi1="  << s.pi1;
        return os;
    }
};

/// HLLC flux for the 6-equation model.
///
/// All components except `u_interface` are face fluxes consistent with
/// the conservative state vector above.  `u_interface` is the HLLC
/// contact-wave speed S_M, used by the non-conservative discretization
/// of the alpha row and the per-phase energy rows (Saurel 2009 lines
/// 836 / 852).
struct Flux
{
    Set::Scalar mass0            = 0.0;  // (alpha_1 rho_1) u
    Set::Scalar mass1            = 0.0;  // (alpha_2 rho_2) u
    Set::Scalar momentum_normal  = 0.0;  // rho u_n^2 + p
    Set::Scalar momentum_tangent = 0.0;  // rho u_n u_t
    Set::Scalar energy_total     = 0.0;  // (rho E + p) u_n (for redundant rho E)
    Set::Scalar energy0          = 0.0;  // (alpha_1 rho_1 e_1) u_n
    Set::Scalar energy1          = 0.0;  // (alpha_2 rho_2 e_2) u_n
    Set::Scalar u_interface      = 0.0;  // HLLC contact speed S_M

    Flux() = default;

    friend std::ostream &
    operator<<(std::ostream &os, const Flux &f)
    {
        os << "m0=" << f.mass0
           << ", m1=" << f.mass1
           << ", Mn=" << f.momentum_normal
           << ", Mt=" << f.momentum_tangent
           << ", E="  << f.energy_total
           << ", E0=" << f.energy0
           << ", E1=" << f.energy1
           << ", u*=" << f.u_interface;
        return os;
    }
};

class FluidRiemann
{
public:
    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref, Set::Scalar small) = 0;
    virtual ~FluidRiemann() = default;
};

}
}
}

#endif
