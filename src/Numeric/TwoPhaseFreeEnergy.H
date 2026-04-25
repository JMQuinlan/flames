#ifndef NUMERIC_TWOPHASEFREEENERGY_H_
#define NUMERIC_TWOPHASEFREEENERGY_H_

#include <AMReX.H>
#include <cmath>
#include "Set/Set.H"
#include "Set/Base.H"

//
// Two-phase Helmholtz free-energy utilities for the fluid–fluid diffuse-
// interface model in Hydro2. Phase 0 is the gas phase (CPG), phase 1 is the
// liquid phase (Tammann/stiffened gas). Naming convention:
//   mu_eta_{0,1}   : molecular (shear) viscosities  — see Hydro2 inputs
//   mu_chem_{0,1}  : chemical potentials of each phase (this header)
//
// Reference: HYDRO2_CH_KORTEWEG_SPEC_20260425.md, Section 1.4.
//
namespace Numeric
{
namespace FreeEnergy
{

// CPG (gas) Helmholtz free energy per unit mass [J/kg].
//   f_CPG(rho,T) = c_v T [1 - ln(T/T_ref)] + R T ln(rho/rho_ref) + q - q' T
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Set::Scalar f_CPG(Set::Scalar rho,    Set::Scalar T,
                  Set::Scalar c_v,    Set::Scalar R,
                  Set::Scalar T_ref,  Set::Scalar rho_ref,
                  Set::Scalar q,      Set::Scalar qprime)
{
    return c_v * T * (Set::Scalar(1.0) - std::log(T / T_ref))
         + R * T * std::log(rho / rho_ref)
         + q - qprime * T;
}

// Tammann (stiffened gas) Helmholtz free energy per unit mass [J/kg].
//   f_T(rho,T) = c_v T [1 - ln(T/T_ref)] + pi/rho + (gamma-1) c_v T ln(rho/rho_ref)
//              + q - q' T
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Set::Scalar f_Tammann(Set::Scalar rho,    Set::Scalar T,
                      Set::Scalar c_v,    Set::Scalar gamma,
                      Set::Scalar pi_0,
                      Set::Scalar T_ref,  Set::Scalar rho_ref,
                      Set::Scalar q,      Set::Scalar qprime)
{
    return c_v * T * (Set::Scalar(1.0) - std::log(T / T_ref))
         + pi_0 / rho
         + (gamma - Set::Scalar(1.0)) * c_v * T * std::log(rho / rho_ref)
         + q - qprime * T;
}

// Per-phase chemical potential (Gibbs free energy per unit mass) [J/kg].
// Definition: mu_chem_k = f_k + p_k / rho_k.
// For CPG:     p_g = rho R T            -> mu_chem_g = f_g + R T
// For Tammann: p_l = (gamma-1) c_v rho T - pi_0
//             -> mu_chem_l = f_l + (gamma-1) c_v T - pi_0/rho_l
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Set::Scalar mu_chem_CPG(Set::Scalar rho,    Set::Scalar T,
                        Set::Scalar c_v,    Set::Scalar R,
                        Set::Scalar T_ref,  Set::Scalar rho_ref,
                        Set::Scalar q,      Set::Scalar qprime)
{
    return f_CPG(rho, T, c_v, R, T_ref, rho_ref, q, qprime) + R * T;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Set::Scalar mu_chem_Tammann(Set::Scalar rho,    Set::Scalar T,
                            Set::Scalar c_v,    Set::Scalar gamma,
                            Set::Scalar pi_0,
                            Set::Scalar T_ref,  Set::Scalar rho_ref,
                            Set::Scalar q,      Set::Scalar qprime)
{
    return f_Tammann(rho, T, c_v, gamma, pi_0, T_ref, rho_ref, q, qprime)
         + (gamma - Set::Scalar(1.0)) * c_v * T
         - pi_0 / rho;
}

// Bulk EOS contribution to dF/d_eta:  f_g(rho_0,T) - f_l(rho_1,T).
// This is the *thermodynamic* part of the CH driving force; it does NOT
// include the lambda*W'(eta) double-well term or the gradient (Korteweg)
// term. Sign convention: positive value drives eta toward 0 (gas).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Set::Scalar dF_deta_bulk(Set::Scalar rho_0,    // gas density
                         Set::Scalar rho_1,    // liquid density
                         Set::Scalar T,
                         // CPG (phase 0) params
                         Set::Scalar cv_0, Set::Scalar R_0,
                         Set::Scalar T_ref_0, Set::Scalar rho_ref_0,
                         Set::Scalar q_0, Set::Scalar qprime_0,
                         // Tammann (phase 1) params
                         Set::Scalar cv_1, Set::Scalar gamma_1, Set::Scalar pi_1,
                         Set::Scalar T_ref_1, Set::Scalar rho_ref_1,
                         Set::Scalar q_1, Set::Scalar qprime_1)
{
    const Set::Scalar fg = f_CPG    (rho_0, T, cv_0, R_0,    T_ref_0, rho_ref_0, q_0, qprime_0);
    const Set::Scalar fl = f_Tammann(rho_1, T, cv_1, gamma_1, pi_1,
                                     T_ref_1, rho_ref_1, q_1, qprime_1);
    return fg - fl;
}

// Latent heat [J/kg] at temperature T. Defined via the standard
//   L = T (s_l - s_g) = (h_l - h_g) at coexistence,
// but evaluated here from the free-energy parameters as
//   L(T) = (q_l - q_g) - T (q'_l - q'_g)
// using the entropy reference offset that fixes the saturation curve slope.
// Useful for offline calibration / Stefan-problem inputs.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Set::Scalar L_vap(Set::Scalar T,
                  Set::Scalar q_g,      Set::Scalar qprime_g,
                  Set::Scalar q_l,      Set::Scalar qprime_l)
{
    return (q_l - q_g) - T * (qprime_l - qprime_g);
}

}} // namespace Numeric::FreeEnergy

#endif // NUMERIC_TWOPHASEFREEENERGY_H_
