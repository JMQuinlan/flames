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

    // Joint two-point Helmholtz calibration.  Solves the 2x2 linear system for
    // (q_l, qprime_l) such that f_g(rho_g_sat, T) = f_l(rho_l_sat, T; q_l, qprime_l)
    // at TWO saturation points.  This is what the CH solver actually wants:
    // dmu_bulk = f_g - f_l = 0 at saturation, AND a non-zero Clausius-Clapeyron
    // slope so that off-saturation conditions produce a physical driving force.
    //
    // The single-point (Calibration above, Gibbs target) zeros mu_chem at one T
    // but leaves qprime_l = 0, killing the saturation curve's T-sensitivity --
    // the model has no Clausius-Clapeyron, dmu_bulk barely responds to superheat,
    // Stefan-problem mass flux is ~20x too small, and one is forced to compensate
    // with an unphysically large mobility.  Use this two-point variant when
    // setting up phase-change problems.
    //
    // Math: with q_g = qprime_g = 0,
    //   q_l - qprime_l * T_k = D_k,    D_k = f_g(rho_g_k, T_k) - f_l_base(rho_l_k, T_k)
    // where f_l_base means f_l evaluated with q_l = qprime_l = 0.  Then
    //   qprime_l = (D_1 - D_2) / (T_2 - T_1)
    //   q_l      = D_1 + qprime_l * T_1
    //
    // Saturation densities provided directly (call once per fluid pair):
    //   point 1 (T_1, rho_g_1, rho_l_1)
    //   point 2 (T_2, rho_g_2, rho_l_2)
    // Returns 0 if both calibrated D_k residuals are < tol; prints (q_l, qprime_l).
    static int JointCalibrationTwoPoint(int verbose = 1)
    {
        // -- Air (CPG, phase 0) ---------------------------------------------
        const Set::Scalar gamma_g   = 1.4;
        const Set::Scalar R_g       = 287.0;
        const Set::Scalar cv_g      = R_g / (gamma_g - 1.0);
        const Set::Scalar rho_g_ref = 1.0;
        const Set::Scalar q_g       = 0.0;
        const Set::Scalar qprime_g  = 0.0;

        // -- Water (Tammann, phase 1) ---------------------------------------
        const Set::Scalar gamma_l   = 4.4;
        const Set::Scalar pi_l      = 6.0e8;
        const Set::Scalar cv_l      = 1816.0;
        const Set::Scalar rho_l_ref = 1000.0;

        // -- Two saturation points (water-air, real saturation curve) -------
        // Gas densities here use the CPG R=287 model (not real steam R=461.5)
        // so the calibration is self-consistent with the simulation's EOS.
        const Set::Scalar T_1     = 373.15;             // 100 C
        const Set::Scalar p_sat_1 = 101325.0;
        const Set::Scalar rho_g_1 = p_sat_1 / (R_g * T_1);  // CPG-consistent
        const Set::Scalar rho_l_1 = 958.4;              // saturated water

        const Set::Scalar T_2     = 423.15;             // 150 C
        const Set::Scalar p_sat_2 = 475800.0;
        const Set::Scalar rho_g_2 = p_sat_2 / (R_g * T_2);
        const Set::Scalar rho_l_2 = 917.0;

        // Use T_1 as the shared reference temperature in f (T_ref slot).
        const Set::Scalar T_ref   = T_1;

        // -- D_k = f_g(rho_g_k, T_k) - f_l(rho_l_k, T_k; q=0, q'=0) ---------
        const Set::Scalar f_g_1 = ::Numeric::FreeEnergy::f_CPG(
            rho_g_1, T_1, cv_g, R_g, T_ref, rho_g_ref, q_g, qprime_g);
        const Set::Scalar f_l_base_1 = ::Numeric::FreeEnergy::f_Tammann(
            rho_l_1, T_1, cv_l, gamma_l, pi_l, T_ref, rho_l_ref, 0.0, 0.0);
        const Set::Scalar D_1 = f_g_1 - f_l_base_1;

        const Set::Scalar f_g_2 = ::Numeric::FreeEnergy::f_CPG(
            rho_g_2, T_2, cv_g, R_g, T_ref, rho_g_ref, q_g, qprime_g);
        const Set::Scalar f_l_base_2 = ::Numeric::FreeEnergy::f_Tammann(
            rho_l_2, T_2, cv_l, gamma_l, pi_l, T_ref, rho_l_ref, 0.0, 0.0);
        const Set::Scalar D_2 = f_g_2 - f_l_base_2;

        // -- Solve 2x2 ------------------------------------------------------
        const Set::Scalar dT       = T_2 - T_1;
        const Set::Scalar qprime_l = (D_1 - D_2) / dT;
        const Set::Scalar q_l      = D_1 + qprime_l * T_1;

        // -- Verify: dmu_bulk should be ~0 at both saturation points --------
        const Set::Scalar dmu_1 = ::Numeric::FreeEnergy::dF_deta_bulk(
            rho_g_1, rho_l_1, T_1,
            cv_g, R_g, T_ref, rho_g_ref, q_g, qprime_g,
            cv_l, gamma_l, pi_l, T_ref, rho_l_ref, q_l, qprime_l);
        const Set::Scalar dmu_2 = ::Numeric::FreeEnergy::dF_deta_bulk(
            rho_g_2, rho_l_2, T_2,
            cv_g, R_g, T_ref, rho_g_ref, q_g, qprime_g,
            cv_l, gamma_l, pi_l, T_ref, rho_l_ref, q_l, qprime_l);

        const Set::Scalar scale = std::max({std::abs(f_g_1), std::abs(f_g_2),
                                            std::abs(f_l_base_1), std::abs(f_l_base_2),
                                            Set::Scalar(1.0)});
        const Set::Scalar tol_rel = 1.0e-10;
        const bool failed = (std::abs(dmu_1)/scale > tol_rel) ||
                            (std::abs(dmu_2)/scale > tol_rel);

        // L_vap diagnostic at T_1 (sanity-check magnitude)
        const Set::Scalar L_T1 = ::Numeric::FreeEnergy::L_vap(T_1, q_g, qprime_g, q_l, qprime_l);

        if (verbose || failed)
        {
            Util::Message(INFO, "  Two-point Helmholtz calibration (q_g = qprime_g = 0):");
            Util::Message(INFO, "    point 1: T = ", T_1, " K, rho_g = ", rho_g_1, ", rho_l = ", rho_l_1);
            Util::Message(INFO, "    point 2: T = ", T_2, " K, rho_g = ", rho_g_2, ", rho_l = ", rho_l_2);
            Util::Message(INFO, "    => q_eta1      = ", q_l);
            Util::Message(INFO, "    => qprime_eta1 = ", qprime_l);
            Util::Message(INFO, "    dmu_bulk(T_1)  = ", dmu_1, " (should be ~0)");
            Util::Message(INFO, "    dmu_bulk(T_2)  = ", dmu_2, " (should be ~0)");
            Util::Message(INFO, "    L_vap(T_1)     = ", L_T1, " J/kg (water ~2.26e6)");
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
