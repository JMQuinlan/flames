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

    /// Get specific heat at constant volume [J/(kg·K)]
    AMREX_GPU_HOST_DEVICE
    virtual Set::Scalar Cv() const = 0;

    /// Get specific heat at constant pressure [J/(kg·K)]
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

    /// Compute mixture sound speed from (rho, p, eta)
    /// Uses: a = sqrt(gamma_eff * (p + p0_eff) / rho)
    /// This is your current formulation from Mix() function
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
