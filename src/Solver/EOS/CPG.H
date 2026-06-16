#ifndef SOLVER_EOS_CPG_H
#define SOLVER_EOS_CPG_H

#include "EOS.H"
#include "IO/ParmParse.H"

namespace Solver
{
namespace EOS
{

/// Calorically Perfect Gas (Ideal Gas) Equation of State
/// Thermodynamic relations:
///   p = (gamma - 1)*rho*e
///   T = e / cv
///   a = sqrt(gamma*p/rho)
class CPG : public EOS
{
private:
    Set::Scalar m_gamma; // Ratio of specific heats (dimensionless)
    Set::Scalar m_cv;    // Specific heat at constant volume [J/(kg·K)]
    Set::Scalar m_cp;    // Specific heat at constant pressure [J/(kg·K)]

public:
    /// Default constructor (air parameters as example)
    CPG()
        : m_gamma(1.4),
          m_cv(717.5),
          m_cp(1.4 * 717.5) // cp = gamma * cv
    {
    }

    /// Constructor with parameters
    /// @param gamma  Ratio of specific heats
    /// @param cv     Specific heat at constant volume [J/(kg·K)]
    /// @param cp     Specific heat at constant pressure [J/(kg·K)]
    CPG(Set::Scalar gamma, Set::Scalar cv, Set::Scalar cp)
        : m_gamma(gamma),
          m_cv(cv),
          m_cp(cp)
    {
    }

    /// Parse from input file with prefix
    /// Reads: prefix.gamma, prefix.cv, prefix.cp
    /// Example input file:
    ///   eos1.gamma = 1.4
    ///   eos1.cv = 717.5
    ///   eos1.cp = 1004.5
    static void Parse(CPG &eos, IO::ParmParse &pp, std::string prefix)
    {
        Set::Scalar gamma, cv, cp;
        pp.query_required(prefix + "gamma", gamma);
        pp.query_required(prefix + "cv", cv);
        pp.query_required(prefix + "cp", cp);
        eos = CPG(gamma, cv, cp);
    }

    // ========================================================================
    // SINGLE-PHASE PROPERTY GETTERS
    // ========================================================================

    AMREX_GPU_HOST_DEVICE
    Set::Scalar Gamma() const override { return m_gamma; }

    AMREX_GPU_HOST_DEVICE
    Set::Scalar P0() const override { return 0.0; } // No stiffening for ideal gas

    AMREX_GPU_HOST_DEVICE
    Set::Scalar Cv() const override { return m_cv; }

    AMREX_GPU_HOST_DEVICE
    Set::Scalar Cp() const override { return m_cp; }

    // ========================================================================
    // SINGLE-PHASE THERMODYNAMIC CALCULATIONS
    // ========================================================================

    /// Compute pressure from (rho, e)
    /// Formula: p = (gamma - 1)*rho*e
    AMREX_GPU_HOST_DEVICE
    Set::Scalar Pressure(Set::Scalar rho, Set::Scalar e) const override
    {
        return (m_gamma - 1.0) * rho * e;
    }

    /// Compute temperature from (rho, e)
    /// Formula: T = e / cv
    AMREX_GPU_HOST_DEVICE
    Set::Scalar Temperature(Set::Scalar rho, Set::Scalar e) const override
    {
        return e / m_cv;
    }

    /// Compute sound speed from (rho, p)
    /// Formula: a = sqrt(gamma*p/rho)
    AMREX_GPU_HOST_DEVICE
    Set::Scalar SoundSpeed(Set::Scalar rho, Set::Scalar p) const override
    {
        return std::sqrt(m_gamma * p / rho);
    }

    /// Compute internal energy from (rho, p)
    /// Formula: e = p / ((gamma-1)*rho)
    AMREX_GPU_HOST_DEVICE
    Set::Scalar InternalEnergy(Set::Scalar rho, Set::Scalar p) const override
    {
        return p / ((m_gamma - 1.0) * rho);
    }
};

} // namespace EOS
} // namespace Solver

#endif
