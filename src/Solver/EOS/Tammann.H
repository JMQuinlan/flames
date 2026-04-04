#ifndef SOLVER_EOS_TAMMANN_H
#define SOLVER_EOS_TAMMANN_H

#include "EOS.H"
#include "IO/ParmParse.H"

namespace Solver
{
namespace EOS
{

/// Tammann (Stiffened Gas) Equation of State
/// Thermodynamic relations:
///   p = (gamma - 1)*rho*e - gamma*p0
///   T = (p + gamma*p0) / ((gamma-1)*rho*cv)
///   a = sqrt(gamma*(p + p0)/rho)
class Tammann : public EOS
{
private:
    Set::Scalar m_gamma; // Ratio of specific heats (dimensionless)
    Set::Scalar m_p0;    // Stiffened gas reference pressure [Pa]
    Set::Scalar m_cv;    // Specific heat at constant volume [J/(kg·K)]
    Set::Scalar m_cp;    // Specific heat at constant pressure [J/(kg·K)]

public:
    /// Default constructor (water parameters as example)
    Tammann()
        : m_gamma(7.15),
          m_p0(3.0e8),
          m_cv(1816.0),
          m_cp(7.15 * 1816.0) // cp = gamma * cv
    {
    }

    /// Constructor with parameters
    /// @param gamma  Ratio of specific heats
    /// @param p0     Stiffened gas reference pressure [Pa]
    /// @param cv     Specific heat at constant volume [J/(kg·K)]
    /// @param cp     Specific heat at constant pressure [J/(kg·K)]
    Tammann(Set::Scalar gamma, Set::Scalar p0, Set::Scalar cv, Set::Scalar cp)
        : m_gamma(gamma),
          m_p0(p0),
          m_cv(cv),
          m_cp(cp)
    {
    }

    /// Parse from input file with prefix
    /// Reads: prefix.gamma, prefix.p0, prefix.cv, prefix.cp
    /// Example input file:
    ///   eos0.gamma = 7.15
    ///   eos0.p0 = 3.0e8
    ///   eos0.cv = 1816.0
    ///   eos0.cp = 12984.4
    static void Parse(Tammann &eos, IO::ParmParse &pp, std::string prefix)
    {
        Set::Scalar gamma, p0, cv, cp;
        pp.query_required(prefix + "gamma", gamma);
        pp.query_required(prefix + "p0", p0);
        pp.query_required(prefix + "cv", cv);
        pp.query_required(prefix + "cp", cp);
        eos = Tammann(gamma, p0, cv, cp);
    }

    // ========================================================================
    // SINGLE-PHASE PROPERTY GETTERS
    // ========================================================================

    AMREX_GPU_HOST_DEVICE
    Set::Scalar Gamma() const override { return m_gamma; }

    AMREX_GPU_HOST_DEVICE
    Set::Scalar P0() const override { return m_p0; }

    AMREX_GPU_HOST_DEVICE
    Set::Scalar Cv() const override { return m_cv; }

    AMREX_GPU_HOST_DEVICE
    Set::Scalar Cp() const override { return m_cp; }

    // ========================================================================
    // SINGLE-PHASE THERMODYNAMIC CALCULATIONS
    // ========================================================================

    /// Compute pressure from (rho, e)
    /// Formula: p = (gamma - 1)*rho*e - gamma*p0
    AMREX_GPU_HOST_DEVICE
    Set::Scalar Pressure(Set::Scalar rho, Set::Scalar e) const override
    {
        return (m_gamma - 1.0) * rho * e - m_gamma * m_p0;
    }

    /// Compute temperature from (rho, e)
    /// Formula: T = (p + gamma*p0) / ((gamma-1)*rho*cv)
    AMREX_GPU_HOST_DEVICE
    Set::Scalar Temperature(Set::Scalar rho, Set::Scalar e) const override
    {
        Set::Scalar p = Pressure(rho, e);
        return (p + m_gamma * m_p0) / ((m_gamma - 1.0) * rho * m_cv);
    }

    /// Compute sound speed from (rho, p)
    /// Formula: a = sqrt(gamma*(p + p0)/rho)
    AMREX_GPU_HOST_DEVICE
    Set::Scalar SoundSpeed(Set::Scalar rho, Set::Scalar p) const override
    {
        return std::sqrt(m_gamma * std::max(p + m_p0,0.0) / rho);
    }

    /// Compute internal energy from (rho, p)
    /// Formula: e = (p + gamma*p0) / ((gamma-1)*rho)
    AMREX_GPU_HOST_DEVICE
    Set::Scalar InternalEnergy(Set::Scalar rho, Set::Scalar p) const override
    {
        return std::max(p + m_gamma * m_p0, 0.0) / ((m_gamma - 1.0) * rho);
    }
};

} // namespace EOS
} // namespace Solver

#endif
