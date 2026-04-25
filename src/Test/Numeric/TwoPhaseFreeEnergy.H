#ifndef TEST_NUMERIC_TWOPHASEFREEENERGY
#define TEST_NUMERIC_TWOPHASEFREEENERGY

#include <cmath>

#include "Set/Set.H"
#include "Numeric/TwoPhaseFreeEnergy.H"

namespace Test
{
namespace Numeric
{

// Phase 1 thermodynamic-consistency check (HYDRO2_CH_KORTEWEG_SPEC §1.4):
// at a chosen reference state (T_ref, rho_g*, rho_l*) the per-phase chemical
// potentials must agree once q_l is calibrated. Calibration:
//   set q_g = q'_g = q'_l = 0; choose q_l = mu_g(rho_g*,T_ref) - mu_l(rho_l*,T_ref).
// After this calibration |mu_g - mu_l| must be at machine precision.
class TwoPhaseFreeEnergy
{
public:
    TwoPhaseFreeEnergy() {}
    ~TwoPhaseFreeEnergy() {}

    // Returns 0 on success, 1 on failure (matches Util::Test::SubMessage convention).
    static int Calibration(int verbose = 0)
    {
        // Air (CPG, phase 0)
        const Set::Scalar gamma_g   = 1.4;
        const Set::Scalar R_g       = 287.0;     // [J/(kg·K)]
        const Set::Scalar cv_g      = R_g / (gamma_g - 1.0); // ~717.5 J/(kg·K)
        const Set::Scalar rho_g_ref = 1.0;       // kg/m^3
        const Set::Scalar rho_g_eq  = 0.6;       // approx air density at 373 K, 1 atm

        // Water (Tammann, phase 1)
        const Set::Scalar gamma_l   = 4.4;
        const Set::Scalar pi_l      = 6.0e8;     // Pa
        const Set::Scalar cv_l      = 1816.0;    // J/(kg·K)
        const Set::Scalar rho_l_ref = 1000.0;
        const Set::Scalar rho_l_eq  = 958.0;     // approx water density at 373 K

        // Shared reference temperature
        const Set::Scalar T_ref = 373.15;

        // Uncalibrated entropy / energy references
        Set::Scalar q_g = 0.0, qprime_g = 0.0;
        Set::Scalar q_l = 0.0, qprime_l = 0.0;

        // Uncalibrated potentials
        const Set::Scalar mu_chem_g_pre = ::Numeric::FreeEnergy::mu_chem_CPG(
            rho_g_eq, T_ref, cv_g, R_g, T_ref, rho_g_ref, q_g, qprime_g);
        const Set::Scalar mu_chem_l_pre = ::Numeric::FreeEnergy::mu_chem_Tammann(
            rho_l_eq, T_ref, cv_l, gamma_l, pi_l, T_ref, rho_l_ref, q_l, qprime_l);

        // Calibrate q_l: shift the liquid free energy so the two potentials match.
        q_l = mu_chem_g_pre - mu_chem_l_pre;

        // Calibrated potentials
        const Set::Scalar mu_chem_g = mu_chem_g_pre;  // unaffected
        const Set::Scalar mu_chem_l = ::Numeric::FreeEnergy::mu_chem_Tammann(
            rho_l_eq, T_ref, cv_l, gamma_l, pi_l, T_ref, rho_l_ref, q_l, qprime_l);

        const Set::Scalar diff  = std::abs(mu_chem_g - mu_chem_l);
        const Set::Scalar scale = std::max({std::abs(mu_chem_g), std::abs(mu_chem_l), Set::Scalar(1.0)});
        const Set::Scalar rel   = diff / scale;

        const Set::Scalar tol = 1.0e-10;
        const bool failed = !(rel < tol);

        if (verbose || failed)
        {
            Util::Message(INFO, "  mu_chem_g  = ", mu_chem_g);
            Util::Message(INFO, "  mu_chem_l  = ", mu_chem_l);
            Util::Message(INFO, "  |diff|     = ", diff);
            Util::Message(INFO, "  relative   = ", rel);
            Util::Message(INFO, "  q_l (cal)  = ", q_l);
        }

        return failed ? 1 : 0;
    }

    // Sanity: f_CPG and f_Tammann are continuous and finite at the reference state.
    static int FiniteAtReference(int /*verbose*/ = 0)
    {
        const Set::Scalar T_ref = 300.0;
        const Set::Scalar f_g = ::Numeric::FreeEnergy::f_CPG(
            1.0, T_ref, 717.5, 287.0, T_ref, 1.0, 0.0, 0.0);
        const Set::Scalar f_l = ::Numeric::FreeEnergy::f_Tammann(
            1000.0, T_ref, 1816.0, 4.4, 6.0e8, T_ref, 1000.0, 0.0, 0.0);
        const bool ok = std::isfinite(f_g) && std::isfinite(f_l);
        return ok ? 0 : 1;
    }
};

} // namespace Numeric
} // namespace Test

#endif
