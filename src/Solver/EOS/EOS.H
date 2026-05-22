#ifndef SOLVER_EOS_EOS_H
#define SOLVER_EOS_EOS_H

#include "AMReX.H"
#include "Set/Set.H"
// NOTE: do NOT add `#include "Util/Util.H"` here -- it causes a dependency
// cycle through Set/Set.H -> ... -> Util.H -> ... -> EOS.H that breaks
// Make's .d generation with a cryptic "must specify -M or -MM" error from
// cc1plus.  Util::Message / Util::Abort are reachable transitively via
// AMReX.H + Set/Set.H in every TU that uses EOS.H in practice; verified.

#ifdef PELE_ENABLED
#include "Solver/EOS/PelePhysicsEOS.H"
#endif

namespace Solver
{
namespace EOS
{

// ============================================================================
// EOS BACKEND SELECTION (input file: eos.backend = tammann | pelephysics)
// ============================================================================
// The static EOS::Phasic* methods below dispatch into either the native
// Tammann/CPG math (default) or the PelePhysicsEOS shim (when built with
// PELE_ENABLED and selected at runtime).  The PelePhysics route is only
// taken on the GAS branch (pi_k == 0) since PelePhysics has no stiffened
// gas; Tammann liquid (pi_k > 0) ALWAYS stays native.
//
// GPU CAVEAT: the runtime check below reads a host-side static variable
// from inside potential ParallelFor lambdas.  Works on CPU only.  When/if
// this codebase grows GPU support (AMREX_USE_CUDA), the dispatch will
// need to move to compile-time (template the EOS choice) or use a device
// constant.  See bin/PeleC.md for the long-term plan.
// ============================================================================
enum class Backend
{
    Native,       // existing Tammann + CPG implementation (default)
    PelePhysics   // gas-branch routes to PelePhysicsEOS shim
};

/// Singleton accessor for the active backend.  Set once at Hydro2::Parse
/// time from the `eos.backend` input parameter.
inline Backend& current_backend()
{
    static Backend b = Backend::Native;
    return b;
}

/// Parse and set from input-file string.  Aborts on unknown names or
/// when "pelephysics" is requested in a non-PELE_ENABLED binary.
inline void SetBackend(const std::string& name)
{
    if (name.empty() || name == "tammann" || name == "native")
    {
        current_backend() = Backend::Native;
        Util::Message(INFO, "EOS backend: native (Tammann + CPG)");
    }
    else if (name == "pelephysics")
    {
#ifdef PELE_ENABLED
        current_backend() = Backend::PelePhysics;
        Util::Message(INFO, "EOS backend: pelephysics (gas branch routed through "
                            "Solver::EOS::PelePhysicsEOS; Tammann liquid stays native)");
#else
        Util::Abort(INFO, "eos.backend=pelephysics requested but this binary was "
                          "not built with PELE_ENABLED.  Rebuild after "
                          "`export PELE_ENABLED=1` (see bin/PeleC.md).");
#endif
    }
    else
    {
        Util::Abort(INFO, "Unknown eos.backend = '", name,
                          "'.  Allowed: tammann | native | pelephysics.");
    }
}

/// Base class for Equation of State
/// Provides interface for single-phase thermodynamics and diffuse interface mixing
class EOS
{
public:
    virtual ~EOS() {}

    // ========================================================================
    // SINGLE-PHASE THERMODYNAMIC PROPERTIES
    // These return properties for a PURE fluid (no mixing)
    // ========================================================================

    /// Get gamma (ratio of specific heats) - constant for each fluid
    AMREX_GPU_HOST_DEVICE
    virtual Set::Scalar Gamma() const = 0;

    /// Get reference pressure p0 (for Tammann: stiffened gas pressure, for CPG: 0)
    AMREX_GPU_HOST_DEVICE
    virtual Set::Scalar P0() const = 0;

    /// Get specific heat at constant volume [J/(kg�K)]
    AMREX_GPU_HOST_DEVICE
    virtual Set::Scalar Cv() const = 0;

    /// Get specific heat at constant pressure [J/(kg�K)]
    AMREX_GPU_HOST_DEVICE
    virtual Set::Scalar Cp() const = 0;

    // ========================================================================
    // SINGLE-PHASE THERMODYNAMIC CALCULATIONS
    // These compute state variables for a PURE fluid
    // ========================================================================

    /// Compute pressure from (rho, e) where e is specific internal energy [J/kg]
    /// For Tammann: p = (gamma-1)*rho*e - gamma*p0
    /// For CPG:     p = (gamma-1)*rho*e
    AMREX_GPU_HOST_DEVICE
    virtual Set::Scalar Pressure(Set::Scalar rho, Set::Scalar e) const = 0;

    /// Compute temperature from (rho, e)
    /// For Tammann: T = (p + gamma*p0) / ((gamma-1)*rho*cv)
    /// For CPG:     T = e / cv
    AMREX_GPU_HOST_DEVICE
    virtual Set::Scalar Temperature(Set::Scalar rho, Set::Scalar e) const = 0;

    /// Compute sound speed from (rho, p)
    /// For Tammann: a = sqrt(gamma*(p + p0)/rho)
    /// For CPG:     a = sqrt(gamma*p/rho)
    AMREX_GPU_HOST_DEVICE
    virtual Set::Scalar SoundSpeed(Set::Scalar rho, Set::Scalar p) const = 0;

    /// Compute internal energy from (rho, p)
    /// For Tammann: e = (p + gamma*p0) / ((gamma-1)*rho)
    /// For CPG:     e = p / ((gamma-1)*rho)
    AMREX_GPU_HOST_DEVICE
    virtual Set::Scalar InternalEnergy(Set::Scalar rho, Set::Scalar p) const = 0;

    // ========================================================================
    // DIFFUSE INTERFACE MIXING FUNCTIONS
    // These compute EFFECTIVE mixture properties from volume fraction eta
    // ========================================================================

    /// Compute mixed gamma using volume-fraction and (gamma-1) weighting
    /// gamma_eff = 1 + 1/A where A = eta/(gamma0-1) + (1-eta)/(gamma1-1)
    /// This is your current formulation from Mix() function
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedGamma(Set::Scalar eta,
                                  const EOS &eos0,
                                  const EOS &eos1)
    {
        Set::Scalar gamma0 = eos0.Gamma();
        Set::Scalar gamma1 = eos1.Gamma();
        Set::Scalar A = eta / (gamma0 - 1.0) + (1.0 - eta) / (gamma1 - 1.0);
        return 1.0 + 1.0 / A;
    }

    /// Compute mixed reference pressure p0_eff
    /// p0_eff = B/(A*gamma_eff) where:
    ///   A = eta/(gamma0-1) + (1-eta)/(gamma1-1)
    ///   B = eta*gamma0*p0_0/(gamma0-1) + (1-eta)*gamma1*p0_1/(gamma1-1)
    /// This is your current formulation from Mix() function
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedP0(Set::Scalar eta,
                               const EOS &eos0,
                               const EOS &eos1)
    {
        Set::Scalar gamma0 = eos0.Gamma();
        Set::Scalar gamma1 = eos1.Gamma();
        Set::Scalar p0_0 = eos0.P0();
        Set::Scalar p0_1 = eos1.P0();

        Set::Scalar A = eta / (gamma0 - 1.0) + (1.0 - eta) / (gamma1 - 1.0);
        Set::Scalar B = (eta * gamma0 * p0_0) / (gamma0 - 1.0)
                        + ((1.0 - eta) * gamma1 * p0_1) / (gamma1 - 1.0);
        Set::Scalar gamma_eff = 1.0 + 1.0 / A;

        return B / (A * gamma_eff);
    }

    /// Compute mixed specific heat at constant volume (linear interpolation)
    /// cv_eff = eta*cv0 + (1-eta)*cv1
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedCv(Set::Scalar eta,
                               const EOS &eos0,
                               const EOS &eos1)
    {
        Set::Scalar A = eta / (eos0.Gamma() - 1.0) + (1.0 - eta) / (eos1.Gamma() - 1.0);
        Set::Scalar B = eta / (eos0.Cv() * (eos0.Gamma() - 1.0)) + (1.0 - eta) / (eos1.Cv() * (eos1.Gamma() - 1.0));
        
        return A / B;
        //return eta * eos0.Cv() + (1.0 - eta) * eos1.Cv();
    }

    /// Compute mixed specific heat at constant pressure (linear interpolation)
    /// cp_eff = eta*cp0 + (1-eta)*cp1
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedCp(Set::Scalar eta,
                               const EOS &eos0,
                               const EOS &eos1)
    {
        return eta * eos0.Cp() + (1.0 - eta) * eos1.Cp();
    }

    // ========================================================================
    // MIXTURE THERMODYNAMIC CALCULATIONS
    // These compute state variables for the MIXTURE using effective properties
    // ========================================================================

    /// Compute mixture pressure from (rho, UE_vol, eta)
    /// Uses: p = (gamma_eff - 1)*UE_vol - gamma_eff*p0_eff + pref
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedPressure(Set::Scalar rho,
                                     Set::Scalar UE_vol,
                                     Set::Scalar eta,
                                     const EOS &eos0,
                                     const EOS &eos1,
                                     Set::Scalar pref = 0.0,
                                     Set::Scalar small = 1e-16)
    {
        Set::Scalar gamma_eff = MixedGamma(eta, eos0, eos1);
        Set::Scalar p0_eff = MixedP0(eta, eos0, eos1);

        Set::Scalar press = (gamma_eff - 1.0) * UE_vol - gamma_eff * p0_eff + pref;
        return std::max(press, small); // Doesn't look right
        //return std::max(press, -p0_eff + small); // Breaks
        //return std::max(press, -p0_eff * gamma_eff + small);// Breaks
    }

    // Compute Tammann Pressure assuming correct mixing
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar TammannPressure(Set::Scalar rho,
                                       Set::Scalar UE_vol,
                                       Set::Scalar gamma_eff,
                                       Set::Scalar p0_eff,
                                       Set::Scalar pref = 0.0,
                                       Set::Scalar small = 1e-16)
    {
        Set::Scalar press = (gamma_eff - 1.0) * UE_vol - gamma_eff * p0_eff + pref;
        return std::max(press, small); // Doesn't look right
        //return std::max(press, -p0_eff + small); // Breaks
        //return std::max(press, -p0_eff * gamma_eff + small); // Breaks
    }

    /// Compute mixture internal energy from (p, eta)
    /// Uses: UE_vol = (p + pref)*A + B where:
    ///   A = eta/(gamma0-1) + (1-eta)/(gamma1-1)
    ///   B = eta*gamma0*p0_0/(gamma0-1) + (1-eta)*gamma1*p0_1/(gamma1-1)
    /// This is your current formulation from Mix() function
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedInternalEnergy(Set::Scalar p,
                                           Set::Scalar eta,
                                           const EOS &eos0,
                                           const EOS &eos1,
                                           Set::Scalar pref = 0.0,
                                           Set::Scalar small = 1e-16)
    {
        Set::Scalar gamma0 = eos0.Gamma();
        Set::Scalar gamma1 = eos1.Gamma();
        Set::Scalar p0_0 = eos0.P0();
        Set::Scalar p0_1 = eos1.P0();

        Set::Scalar A = eta / (gamma0 - 1.0) + (1.0 - eta) / (gamma1 - 1.0);
        Set::Scalar B = (eta * gamma0 * p0_0) / (gamma0 - 1.0)
                        + ((1.0 - eta) * gamma1 * p0_1) / (gamma1 - 1.0);

        Set::Scalar ue_vol = (p + pref) * A + B;
        return std::max(ue_vol, small);
        // return ue_vol;
    }

    /// Compute mixture temperature from (rho, p, eta)
    /// Uses: T = (p + p0_eff) / (rho * cv_eff * (gamma_eff - 1))
    /// This is your current formulation from RHS() function
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedTemperature(Set::Scalar rho,
                                        Set::Scalar p,
                                        Set::Scalar eta,
                                        const EOS &eos0,
                                        const EOS &eos1,
                                        Set::Scalar pref = 0.0)
    {
        Set::Scalar gamma_eff = MixedGamma(eta, eos0, eos1);
        Set::Scalar p0_eff = MixedP0(eta, eos0, eos1);
        Set::Scalar cv_eff = MixedCv(eta, eos0, eos1);

        return (p + gamma_eff * p0_eff) / (rho * cv_eff * (gamma_eff - 1.0));
    }

    /// Compute mixture sound speed from (rho, p, eta) -- ORIGINAL FROZEN FORM.
    /// Uses: a = sqrt(gamma_eff * (p + p0_eff) / rho)
    /// This treats the diffuse band as a single Tammann fluid with effective
    /// gamma and p0. It does NOT exhibit the two-phase sound-speed dip that
    /// the true mechanical-equilibrium mixture has at intermediate alpha.
    /// Kept here for reference / fallback; superseded by the Wood-form below.
    /*
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedSoundSpeed(Set::Scalar rho,
                                       Set::Scalar p,
                                       Set::Scalar eta,
                                       const EOS &eos0,
                                       const EOS &eos1,
                                       Set::Scalar pref = 0.0,
                                       Set::Scalar small = 1e-16)
    {
        Set::Scalar gamma_eff = MixedGamma(eta, eos0, eos1);
        Set::Scalar p0_eff = MixedP0(eta, eos0, eos1);

        Set::Scalar mixed_press = (p + p0_eff);
        if (mixed_press <= 0.0)
        {
            return small;
        }
        else
        {
            return std::sqrt(gamma_eff * (p + p0_eff) / rho);
        }
    }
    */

    /// Compute mixture sound speed from (rho, p, eta) -- Wood (Kapila) form.
    ///
    /// Bulk-modulus mixing rule for a mechanical-equilibrium two-phase mixture:
    ///     1/(rho_mix * a_W^2) = alpha / (rho_0 a_0^2) + (1-alpha) / (rho_1 a_1^2)
    /// For Tammann pure phase k:  rho_k a_k^2 = gamma_k (p + pi_k), so
    ///     1/(rho_mix * a_W^2) = alpha / (gamma_0 (p + pi_0))
    ///                         + (1-alpha) / (gamma_1 (p + pi_1))
    ///
    /// Reduces to the pure-phase Tammann sound speed in the alpha->0 / alpha->1
    /// limits and produces the well-known sound-speed dip at intermediate alpha
    /// (Wood 1930; Saurel-Petitpas-Berry 2009 Fig. 2).
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedSoundSpeed(Set::Scalar rho,
                                       Set::Scalar p,
                                       Set::Scalar eta,
                                       const EOS &eos0,
                                       const EOS &eos1,
                                       Set::Scalar pref = 0.0,
                                       Set::Scalar small = 1e-16)
    {
        (void)pref;

        Set::Scalar gamma0 = eos0.Gamma();
        Set::Scalar gamma1 = eos1.Gamma();
        Set::Scalar p0_0   = eos0.P0();
        Set::Scalar p0_1   = eos1.P0();

        // Per-phase (p + pi); guard against unphysical (p + pi) <= 0.
        Set::Scalar pp_0 = p + p0_0;
        Set::Scalar pp_1 = p + p0_1;
        if (pp_0 <= 0.0 || pp_1 <= 0.0)
        {
            return small;
        }

        // Per-phase compressibilities 1/(rho_k a_k^2) = 1/(gamma_k (p + pi_k))
        Set::Scalar invK0 = 1.0 / (gamma0 * pp_0);
        Set::Scalar invK1 = 1.0 / (gamma1 * pp_1);

        Set::Scalar invK_mix = eta * invK0 + (1.0 - eta) * invK1;

        if (rho <= small || invK_mix <= 0.0)
        {
            return small;
        }

        // a_W^2 = 1 / (rho_mix * invK_mix)
        return std::sqrt(1.0 / (rho * invK_mix));
    }

    /// Kapila interfacial-relaxation coefficient K for the augmented
    /// 5-equation alpha advection (Schmidmayer-Bryngelson-Colonius JCP 2020,
    /// Eqs. 10/11; equivalent to Beig-Johnsen 2018 Eq. 2.4):
    ///
    ///     d(alpha_1)/dt + u . grad(alpha_1) = K div(u)
    ///
    ///     K = (rho_2 a_2^2 - rho_1 a_1^2)
    ///         / ( rho_1 a_1^2 / alpha_1 + rho_2 a_2^2 / alpha_2 )
    ///
    /// Per-phase Tammann/ideal-gas bulk modulus rho_k * a_k^2 = gamma_k (p + pi_k)
    /// (rho cancels analytically, so we only need EOS constants and pressure).
    ///
    /// Convention: eta = alpha for phase 0 (matches MixedSoundSpeed signature),
    ///   so phase 0 plays the role of Schmidmayer's "phase 1" and phase 1 plays
    ///   his "phase 2". The 1/alpha singularities at alpha->0 or alpha->1 are
    ///   regularized by flooring alpha and (1-alpha) at small; the denominator
    ///   then diverges and K -> 0 in the pure-phase limits, which is the
    ///   correct physical limit (no interfacial relaxation away from the band).
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar KapilaK(Set::Scalar p,
                               Set::Scalar eta,
                               const EOS &eos0,
                               const EOS &eos1,
                               Set::Scalar small = 1e-16)
    {
        Set::Scalar gamma0 = eos0.Gamma();
        Set::Scalar gamma1 = eos1.Gamma();
        Set::Scalar p0_0   = eos0.P0();
        Set::Scalar p0_1   = eos1.P0();

        // rho_k a_k^2 = gamma_k (p + pi_k), floored to keep things finite
        // if p drifts into the unphysical region under a strong rarefaction.
        Set::Scalar rhoa2_0 = gamma0 * std::max(p + p0_0, small);
        Set::Scalar rhoa2_1 = gamma1 * std::max(p + p0_1, small);

        // Floor alpha and (1-alpha) so the 1/alpha denominator stays finite;
        // K -> 0 in those limits naturally.
        Set::Scalar a0 = std::max(eta,         small);
        Set::Scalar a1 = std::max(1.0 - eta,   small);

        Set::Scalar numerator   = rhoa2_1 - rhoa2_0;
        Set::Scalar denominator = rhoa2_0 / a0 + rhoa2_1 / a1;

        return numerator / denominator;
    }

    /// Safe sound speed calculation with error checking
    /// Prevents NaN from negative arguments or division by zero
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar TammannSoundSpeed(Set::Scalar rho,
                                         Set::Scalar p,
                                         Set::Scalar gamma_eff,
                                         Set::Scalar p0_eff,
                                         Set::Scalar small = 1e-16)
    {
        rho = std::max(small, rho);

        Set::Scalar mixed_press = (p + p0_eff);
        if (mixed_press <= 0.0)
        {
            return small;
        }
        else
        {
            return std::sqrt(gamma_eff * (p + p0_eff) / rho);
        }
    }

    // ========================================================================
    // SIX-EQUATION DIFFUSE-INTERFACE HELPERS (Saurel-Petitpas-Berry 2009 /
    // Schmidmayer-Bryngelson-Colonius 2020). All static, AMReX-GPU-safe.
    // Each helper documents the source equation in a one-line tag.
    // Symbol key:
    //   alpha   = alpha_k  (volume fraction of phase k)
    //   arh     = (alpha rho)_k  (canonical conservative phase mass)
    //   E_k     = (alpha rho e)_k  (canonical conservative per-phase internal energy)
    //   gam     = gamma_k         (per-phase adiabatic index)
    //   pi_k    = stiffened-gas reference pressure (Tammann pi_k = P0())
    //   Z_k     = rho_k c_k       (acoustic impedance)
    // ========================================================================

    /// Per-phase pressure from the canonical conservative (alpha rho e)_k.
    /// Saurel-Petitpas-Berry 2009 eq. (II.3):  p_k = (gamma_k - 1) rho_k e_k - gamma_k pi_k.
    /// Substituting rho_k e_k = E_k / alpha gives:
    ///     p_k = (gamma_k - 1) * E_k / alpha  -  gamma_k pi_k
    /// Floor: clamp p_k so that p_k + pi_k >= small (keeps c_k^2 = gamma_k(p+pi)/rho real).
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar PhasicPressureFromEnergy(Set::Scalar E_k,
                                                Set::Scalar alpha,
                                                Set::Scalar gam,
                                                Set::Scalar pi_k,
                                                Set::Scalar small = 1e-16)
    {
#ifdef PELE_ENABLED
        // Gas branch (pi_k == 0) -> PelePhysics shim when selected.
        // Liquid branch (pi_k > 0) always stays native (PelePhysics has no SG).
        if (current_backend() == Backend::PelePhysics && pi_k == 0.0)
        {
            return PelePhysicsEOS::PhasicPressureFromEnergy(E_k, alpha, gam, pi_k, small);
        }
#endif
        Set::Scalar a_safe = std::max(alpha, small);
        Set::Scalar p_k = (gam - 1.0) * (E_k / a_safe) - gam * pi_k;
        // Keep p_k + pi_k > 0 (Saurel 2009 §3 stiffened-gas positivity).
        return std::max(p_k, -pi_k + small);
    }

    /// Canonical conservative per-phase internal energy from per-phase pressure.
    /// Saurel-Petitpas-Berry 2009 eq. (III.5) / Schmidmayer-Bryngelson-Colonius 2020
    /// post-reinit reset:  (alpha rho e)_k = alpha_k (p + gamma_k pi_k) / (gamma_k - 1).
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar PhasicEnergyFromPressure(Set::Scalar p_k,
                                                Set::Scalar alpha,
                                                Set::Scalar gam,
                                                Set::Scalar pi_k,
                                                Set::Scalar small = 1e-16)
    {
#ifdef PELE_ENABLED
        if (current_backend() == Backend::PelePhysics && pi_k == 0.0)
        {
            return PelePhysicsEOS::PhasicEnergyFromPressure(p_k, alpha, gam, pi_k, small);
        }
#endif
        Set::Scalar numer = std::max(p_k + gam * pi_k, small);
        return alpha * numer / (gam - 1.0);
    }

    /// Per-phase Tammann/stiffened-gas sound speed in PURE-PHASE quantities.
    /// c_k = sqrt( gamma_k (p_k + pi_k) / rho_k )
    /// Saurel-Petitpas-Berry 2009 just above eq. (III.2) (per-phase form).
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar PhasicSoundSpeed(Set::Scalar rho_k,
                                        Set::Scalar p_k,
                                        Set::Scalar gam,
                                        Set::Scalar pi_k,
                                        Set::Scalar small = 1e-16)
    {
#ifdef PELE_ENABLED
        if (current_backend() == Backend::PelePhysics && pi_k == 0.0)
        {
            return PelePhysicsEOS::PhasicSoundSpeed(rho_k, p_k, gam, pi_k, small);
        }
#endif
        Set::Scalar r = std::max(rho_k, small);
        Set::Scalar pp = std::max(p_k + pi_k, small);
        return std::sqrt(gam * pp / r);
    }

    /// Frozen mixture sound speed used by the 6-eq hyperbolic step.
    /// Schmidmayer-Bryngelson-Colonius 2020 eq. (17) /
    /// Saurel-Petitpas-Berry 2009 eq. (III.2):
    ///     c^2 = Y_1 c_1^2 + Y_2 c_2^2          (Y_k = mass fraction)
    /// MONOTONIC in alpha - the central robustness lever vs the Wood form.
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar FrozenMixtureSoundSpeed(Set::Scalar Y0,
                                               Set::Scalar Y1,
                                               Set::Scalar c0,
                                               Set::Scalar c1)
    {
        return std::sqrt(Y0 * c0 * c0 + Y1 * c1 * c1);
    }

    /// Interfacial pressure (acoustic-impedance-weighted).
    /// Schmidmayer 2020 eq. (14) / Saurel 2009 just below eq. (II.6):
    ///     p_I = (Z_1 p_0 + Z_0 p_1) / (Z_0 + Z_1),  Z_k = rho_k c_k.
    /// Note the cross-pairing of subscripts (Z_1 multiplies p_0).
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar InterfacialPressureZ(Set::Scalar p0,
                                            Set::Scalar p1,
                                            Set::Scalar Z0,
                                            Set::Scalar Z1,
                                            Set::Scalar small = 1e-16)
    {
        Set::Scalar denom = std::max(Z0 + Z1, small);
        return (Z1 * p0 + Z0 * p1) / denom;
    }

    /// HLLC per-phase Hugoniot pressure star-state.
    /// Saurel-Petitpas-Berry 2009 eq. (II.8) inverted for SG/Tammann (lines ~812):
    ///   p_k*(rho_k*) = (p_k + pi_k) * [(gam+1) rho_k* - (gam-1) rho_k]
    ///                / [(gam+1) rho_k  - (gam-1) rho_k*]   - pi_k
    /// Used to get the per-phase internal energy across the acoustic star
    /// (the volume fraction alpha_k is constant across acoustic waves -
    /// Saurel 2009 line 798).
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar PhasicHugoniotPressure(Set::Scalar rho_star,
                                              Set::Scalar rho,
                                              Set::Scalar p,
                                              Set::Scalar gam,
                                              Set::Scalar pi_k,
                                              Set::Scalar small = 1e-16)
    {
        Set::Scalar gp1 = gam + 1.0;
        Set::Scalar gm1 = gam - 1.0;
        // Cap rho_star at 0.99 * (gamma+1)/(gamma-1) * rho to avoid the
        // strong-shock pole at the asymptotic compression ratio.  Without
        // this, p_star blows up when rho_star is close to the asymptote.
        Set::Scalar rho_max = 0.99 * gp1 / std::max(gm1, small) * std::max(rho, small);
        Set::Scalar rho_s   = std::min(std::max(rho_star, small), rho_max);
        Set::Scalar numer   = gp1 * rho_s - gm1 * rho;
        Set::Scalar denom   = gp1 * rho   - gm1 * rho_s;
        if (std::abs(denom) < small) return p; // fall back to isentropic
        return (p + pi_k) * (numer / denom) - pi_k;
    }

    /// Relaxation specific-volume (Saurel-Petitpas-Berry 2009 eq. (III.4),
    /// frozen-p_hat_I form): v_k(p) = v_k^0 (p_k^0 + gam pi + (gam-1) pHat)
    ///                                / (p     + gam pi + (gam-1) pHat).
    /// Returns v_k(p). Used inside the Newton iteration on
    /// Schmidmayer 2020 eq. (24):  sum_k (alpha rho)_k v_k(p) = 1.
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar RelaxationVolume_SG(Set::Scalar p,
                                           Set::Scalar p_k0,
                                           Set::Scalar rho_k0,
                                           Set::Scalar gam,
                                           Set::Scalar pi_k,
                                           Set::Scalar pHat,
                                           Set::Scalar small = 1e-16)
    {
        Set::Scalar v_k0 = 1.0 / std::max(rho_k0, small);
        Set::Scalar denom = p + gam * pi_k + (gam - 1.0) * pHat;
        denom = std::max(denom, small);
        Set::Scalar numer = p_k0 + gam * pi_k + (gam - 1.0) * pHat;
        return v_k0 * numer / denom;
    }

    /// Derivative of RelaxationVolume_SG with respect to p (for Newton-Raphson).
    /// d v_k / dp = - v_k(p) / [p + gam pi + (gam-1) pHat]
    /// (Direct differentiation of eq. (III.4) in p.)
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar RelaxationVolume_SG_dvdp(Set::Scalar p,
                                                Set::Scalar p_k0,
                                                Set::Scalar rho_k0,
                                                Set::Scalar gam,
                                                Set::Scalar pi_k,
                                                Set::Scalar pHat,
                                                Set::Scalar small = 1e-16)
    {
        Set::Scalar v_k = RelaxationVolume_SG(p, p_k0, rho_k0, gam, pi_k, pHat, small);
        Set::Scalar denom = p + gam * pi_k + (gam - 1.0) * pHat;
        denom = std::max(denom, small);
        return -v_k / denom;
    }

    /// Self-consistent relaxation specific-volume from Schmidmayer-Bryngelson-
    /// Colonius 2020 eq. (25):
    ///     v_k(p) = v_k^0 * (p_k^0 + gam_k pi_k + (gam_k - 1) p) / (gam_k (p + pi_k)).
    /// This is the same as Saurel 2009 eq. (III.4) substituted with p_hat_I = p
    /// (the self-consistent / "relaxed" choice).  Used inside the Newton on
    /// Schmidmayer 2020 eq. (24):  sum_k (alpha rho)_k v_k(p) = 1.
    /// Saurel 2009 §4.4 calls this the "negligibly different" alternative to
    /// the frozen-p_hat_I form; in practice they agree to 4+ digits but are
    /// not bit-identical for high pressure ratios.
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar RelaxationVolume_SG_SC(Set::Scalar p,
                                              Set::Scalar p_k0,
                                              Set::Scalar rho_k0,
                                              Set::Scalar gam,
                                              Set::Scalar pi_k,
                                              Set::Scalar small = 1e-16)
    {
        Set::Scalar v_k0  = 1.0 / std::max(rho_k0, small);
        Set::Scalar numer = p_k0 + gam * pi_k + (gam - 1.0) * p;
        Set::Scalar denom = gam * (p + pi_k);
        denom = std::max(denom, small);
        return v_k0 * numer / denom;
    }

    /// Total derivative of RelaxationVolume_SG_SC with respect to p
    /// (Schmidmayer 2020 eq. 25, self-consistent).  Direct differentiation:
    ///     dv/dp = -v_k^0 (p_k^0 + pi_k) / (gam_k (p + pi_k)^2).
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar RelaxationVolume_SG_SC_dvdp(Set::Scalar p,
                                                   Set::Scalar p_k0,
                                                   Set::Scalar rho_k0,
                                                   Set::Scalar gam,
                                                   Set::Scalar pi_k,
                                                   Set::Scalar small = 1e-16)
    {
        Set::Scalar v_k0  = 1.0 / std::max(rho_k0, small);
        Set::Scalar denom = (p + pi_k);
        denom = std::max(std::abs(denom), small) * (denom >= 0.0 ? 1.0 : -1.0);
        return -v_k0 * (p_k0 + pi_k) / (gam * denom * denom);
    }

    /// Saurel-Petitpas-Berry 2009 §5.3 Fig. 21 heat-exchange curve fit:
    ///     mu(v) = exp(a v^2 + b v + c)
    /// Fig. 21 caption gives, for epoxy-spinel,
    ///     a = -5.64e7, b = 5.34e3, c = -25.6
    /// in the specific-volume window v in [2.65e-4, 4.61e-4] m^3/kg.
    /// Originally calibrated for the Sau09 q*@u/@x form of the AHE source.
    /// In flames2 we repurpose it inside the Schmidmayer-Bryngelson-Colonius
    /// 2020 eq. 13 r-source (-/+ mu p_I Dp on per-phase internal energies);
    /// the magnitudes are not transferable without re-calibration but it is
    /// a defensible starting point.  Sch20 itself does NOT specify a finite
    /// mu - they take the stiff limit mu -> infinity inside the
    /// operator-split Newton on eq. 24.
    ///
    /// v is clamped to [v_min, v_max] before evaluation: outside that window
    /// the quadratic explodes (Fig. 21 only certifies the fit inside the
    /// epoxy-spinel piston-velocity sweep that produced it).
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar AHE_mu_Saurel(Set::Scalar v,
                                     Set::Scalar a,
                                     Set::Scalar b,
                                     Set::Scalar c,
                                     Set::Scalar v_min,
                                     Set::Scalar v_max)
    {
        Set::Scalar v_clamped = std::min(std::max(v, v_min), v_max);
        return std::exp(a * v_clamped * v_clamped + b * v_clamped + c);
    }

    /// Reinitialization mixture pressure from the redundant total energy.
    /// Schmidmayer 2020 eq. (26) / Saurel 2009 eq. (III.5):
    ///     p = ( rho e  - sum_k alpha_k gam_k pi_k / (gam_k - 1) )
    ///         / sum_k ( alpha_k / (gam_k - 1) )
    /// where rho e = rho E - 0.5 rho |u|^2 is the (conserved) mixture internal
    /// energy from the redundant ROE-var. Uses POST-RELAXATION alpha_k.
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar ReinitMixturePressure(Set::Scalar rho_e,
                                             Set::Scalar alpha0,
                                             Set::Scalar alpha1,
                                             Set::Scalar gam0,
                                             Set::Scalar pi0,
                                             Set::Scalar gam1,
                                             Set::Scalar pi1,
                                             Set::Scalar small = 1e-16)
    {
        Set::Scalar A = alpha0 / (gam0 - 1.0) + alpha1 / (gam1 - 1.0);
        Set::Scalar B = alpha0 * gam0 * pi0 / (gam0 - 1.0)
                      + alpha1 * gam1 * pi1 / (gam1 - 1.0);
        return (rho_e - B) / std::max(A, small);
    }

};

} // namespace EOS
} // namespace Solver

#endif
