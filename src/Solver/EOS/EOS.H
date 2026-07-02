#ifndef SOLVER_EOS_EOS_H
#define SOLVER_EOS_EOS_H

#include "AMReX.H"
#include "Set/Set.H"

namespace Solver
{
namespace EOS
{

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
    /// For Tammann: T = (p + p0) / ((gamma-1)*rho*cv)   (textbook SG, q=0; e = cv*T + p0/rho)
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

    /// Phase-dependent pressure floor.  Gas (eta -> 1, p0_eff -> 0) cannot
    /// sustain tension: p >= eps_p.  The stiffened liquid (eta -> 0) may be
    /// pulled into tension down to a cavitation limit -p_cav.  Blend linearly in
    /// eta.  A hyperbolicity guard caps the tension at 0.75*p0_eff so
    /// p + p0_eff >= 0.25*p0_eff (c >= 0.5 c_ref); p -> -p0_eff makes
    /// c^2 = gamma*(p+p0_eff)/rho -> 0, which is why an unguarded -p0_eff floor
    /// breaks the solver.  Defaults (eps_p=p_cav=0) give a vacuum floor.
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar PressureFloor(Set::Scalar eta, Set::Scalar p0_eff,
                                     Set::Scalar eps_p = 0.0, Set::Scalar p_cav = 0.0)
    {
        const Set::Scalar e   = (eta < 0.0) ? 0.0 : ((eta > 1.0) ? 1.0 : eta);
        const Set::Scalar pf  = e * eps_p - (1.0 - e) * p_cav; // gas:+eps_p, liquid:-p_cav
        const Set::Scalar hyp = -0.75 * p0_eff;                // hyperbolicity guard
        return (pf > hyp) ? pf : hyp;
    }

    /// Compute mixture pressure from (rho, UE_vol, eta)
    /// Uses: p = (gamma_eff - 1)*UE_vol - gamma_eff*p0_eff
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedPressure(Set::Scalar rho,
                                     Set::Scalar UE_vol,
                                     Set::Scalar eta,
                                     const EOS &eos0,
                                     const EOS &eos1,
                                     Set::Scalar small = 1e-16,
                                     Set::Scalar eps_p = 0.0,
                                     Set::Scalar p_cav = 0.0)
    {
        Set::Scalar gamma_eff = MixedGamma(eta, eos0, eos1);
        Set::Scalar p0_eff = MixedP0(eta, eos0, eos1);

        Set::Scalar press = (gamma_eff - 1.0) * UE_vol - gamma_eff * p0_eff;
        // Phase-dependent floor: gas p >= eps_p, liquid p >= -p_cav (hyperbolicity
        // guarded). The old uniform max(press, small) clipped legitimate liquid
        // tension to ~0; this lets the bounded tension through to plots/NSCBC.
        return std::max(press, PressureFloor(eta, p0_eff, eps_p, p_cav));
    }

    // Compute Tammann Pressure assuming correct mixing
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar TammannPressure(Set::Scalar rho,
                                       Set::Scalar UE_vol,
                                       Set::Scalar gamma_eff,
                                       Set::Scalar p0_eff,
                                       Set::Scalar small = 1e-16)
    {
        Set::Scalar press = (gamma_eff - 1.0) * UE_vol - gamma_eff * p0_eff;
        return std::max(press, small); // Doesn't look right
        //return std::max(press, -p0_eff + small); // Breaks
        //return std::max(press, -p0_eff * gamma_eff + small); // Breaks
    }

    /// Compute mixture internal energy from (p, eta)
    /// Uses: UE_vol = p*A + B where:
    ///   A = eta/(gamma0-1) + (1-eta)/(gamma1-1)
    ///   B = eta*gamma0*p0_0/(gamma0-1) + (1-eta)*gamma1*p0_1/(gamma1-1)
    /// This is your current formulation from Mix() function
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedInternalEnergy(Set::Scalar p,
                                           Set::Scalar eta,
                                           const EOS &eos0,
                                           const EOS &eos1,
                                           Set::Scalar small = 1e-16)
    {
        Set::Scalar gamma0 = eos0.Gamma();
        Set::Scalar gamma1 = eos1.Gamma();
        Set::Scalar p0_0 = eos0.P0();
        Set::Scalar p0_1 = eos1.P0();

        Set::Scalar A = eta / (gamma0 - 1.0) + (1.0 - eta) / (gamma1 - 1.0);
        Set::Scalar B = (eta * gamma0 * p0_0) / (gamma0 - 1.0)
                        + ((1.0 - eta) * gamma1 * p0_1) / (gamma1 - 1.0);

        Set::Scalar ue_vol = p * A + B;
        return std::max(ue_vol, small);
        // return ue_vol;
    }

    /// Compute mixture temperature from (rho, p, eta) -- THERMAL-EQUILIBRIUM form.
    ///
    /// T = [ eta*(p+pi_0)/((g0-1) cv0) + (1-eta)*(p+pi_1)/((g1-1) cv1) ] / rho
    ///
    /// Derivation: this model builds the mixture density by the volume-fraction
    /// (arithmetic) blend  rho = eta*rho_0 + (1-eta)*rho_1  (see Hydro2 Mix()).
    /// Treat each phase as a pure fluid at the shared pressure p and a COMMON
    /// temperature T, so its intrinsic density is rho_k = (p+pi_k)/((g_k-1) cv_k T)
    /// (textbook stiffened gas). Substituting and solving for T gives the formula
    /// above. It uses the PER-PHASE (pure) gamma/cv/pi, NOT the mixed-effective
    /// values, so across the diffuse band T stays BOUNDED between the two pure-phase
    /// temperatures -- no overshoot. The old single-effective-fluid form
    /// (p + p0_eff)/((g_eff-1) rho cv_eff) divides by the band density, which
    /// collapses toward the gas value while p0_eff lingers, superheating the
    /// interface (hundreds of K) and clamping the saturation closure. This is the
    /// single-density analog of the full two-state model's per-phase thermal-T blend.
    /// Pure limits: gas (pi0=0) -> p/((g0-1) cv0 rho); liquid -> (p+pi1)/((g1-1) cv1 rho).
    AMREX_GPU_HOST_DEVICE
    static Set::Scalar MixedTemperature(Set::Scalar rho,
                                        Set::Scalar p,
                                        Set::Scalar eta,
                                        const EOS &eos0,
                                        const EOS &eos1)
    {
        const Set::Scalar small = 1.0e-16;
        const Set::Scalar g0 = eos0.Gamma(), pi0 = eos0.P0(), cv0 = eos0.Cv();
        const Set::Scalar g1 = eos1.Gamma(), pi1 = eos1.P0(), cv1 = eos1.Cv();
        const Set::Scalar t0 = eta         * std::max(p + pi0, 0.0) / ((g0 - 1.0) * cv0);
        const Set::Scalar t1 = (1.0 - eta) * std::max(p + pi1, 0.0) / ((g1 - 1.0) * cv1);
        return (t0 + t1) / std::max(rho, small);
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
                                       Set::Scalar small = 1e-16)
    {
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
};

} // namespace EOS
} // namespace Solver

#endif
