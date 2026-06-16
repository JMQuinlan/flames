// =============================================================================
// Solver::EOS::PelePhysicsEOS  -- shim layer wrapping PelePhysics' EOS for
// the GAS-PHASE branch of Hydro2.
//
// PHILOSOPHY
//   The Hydro2 6-eq framework already has a Tammann EOS for the liquid
//   phase (Solver::EOS::Tammann) and a CPG / ideal-gas formula for the
//   gas phase, both exposed through static methods on Solver::EOS::EOS.
//   PelePhysics provides a richer gas-phase EOS (NASA-7 polynomial thermo,
//   multi-species mixture rules, eventually SRK for high-pressure) but
//   does NOT support stiffened-gas.
//
//   This shim:
//     - REPLACES the gas-phase branch of EOS::Mixed* / EOS::Phasic*
//       with PelePhysics::eos::EosType calls.
//     - LEAVES the liquid-phase Tammann branch alone.
//     - Matches the static-method signatures of Solver::EOS::EOS so the
//       integrator can dispatch with minimal change.
//     - Compiles only when PELE_ENABLED is set in the build environment;
//       otherwise this header is empty.
//
// PHASE 2 SCOPE (current)
//   - GammaLaw EOS (NUM_SPECIES = 1), so behavior IS identical to a
//     well-tuned CPG.  The purpose of Phase 2 is to prove the integration
//     mechanically without changing physics.  Phase 3 extends to
//     multi-species; Phase 4 adds chemistry.
//
// PHASE 3+ TODO  (this file will gain methods)
//   - Per-species Y_k[] input/output arrays in the API
//   - PressureFromRYE(R, Y, e)        — multi-species pressure
//   - TemperatureFromEY(e, Y)         — multi-species temperature
//   - SoundSpeedFromRPY(R, p, Y)      — mixture sound speed
//   - Enthalpy h_k per species (for chemistry source coupling)
//   - Switch USE_GAMMALAW_EOS -> USE_FUEGO_EOS in the Makefile and pick
//     a real CHEMKIN mechanism (start with H2/air dryer-spc)
//
// See bin/PeleC.md for the full multi-phase roadmap.
// =============================================================================

#ifndef SOLVER_EOS_PELEPHYSICSEOS_H
#define SOLVER_EOS_PELEPHYSICSEOS_H

#ifdef PELE_ENABLED

#include <AMReX.H>
#include <AMReX_REAL.H>
#include "Set/Base.H"
#include "Set/Set.H"

// PelePhysics public headers.  We intentionally pull GammaLaw.H directly
// rather than PelePhysics's EOS.H, for two reasons:
//   (1) PelePhysics ships an EOS.H that collides by name with flames2's own
//       src/Solver/EOS/EOS.H -- and the "" include resolution rule prefers
//       the local file, so #include "EOS.H" inside this shim picks the
//       wrong header.  GammaLaw.H is uniquely named.
//   (2) PelePhysics's EOS.H pulls in mechanism-table functions (CKSYME_STR,
//       CKAWT, ...) that would force us to compile and link the mechanism's
//       .cpp file.  We don't need them for Phase 2 (single-EOS GammaLaw),
//       so dodge the linkage cost.
// Mechanism choice (default = air, 2 species) is set via the -isystem path
// to Mechanisms/<name>/ in the Makefile.  EOS choice is via -DUSE_GAMMALAW_EOS.
#include "mechanism.H"
#include "PhysicsConstants.H"

// GammaLaw.H references EosParm<GammaLaw> in its inline constructor body
// (`eparm->gamma`), so a forward decl alone won't compile -- the body
// needs the FULL type.  But PelePhysics's own EosParams.H has, on master
// against AMReX 25.04, an unrelated incompatibility in the Manifold-EOS
// specialization (`pp.get(name, chemSpecies_Mwt[i])` against the wrong
// AMReX::ParmParse overload).  Dodge both by supplying the SAME minimal
// EosParm<GammaLaw> definition that PelePhysics ships (one field, gamma)
// ourselves.  This avoids touching the Manifold path entirely.  If a
// future PelePhysics version grows the GammaLaw EosParm, mirror it here.
// (Source of truth: ext/pelephysics/Source/Eos/EosParams.H lines 27-31.)
namespace pele::physics::eos {
struct GammaLaw;
template <typename EOSType> struct EosParm;
template <>
struct EosParm<GammaLaw>
{
    amrex::Real gamma { Constants::gamma };
};
}

#include "GammaLaw.H"

namespace Solver
{
namespace EOS
{

/// Static wrapper around pele::physics::eos::EosType providing the same
/// per-phase static API as Solver::EOS::EOS, but routed through PelePhysics.
/// All methods are AMREX_GPU_HOST_DEVICE so they can be called inside
/// ParallelFor lambdas just like the native EOS class.
struct PelePhysicsEOS
{
    // Skipping pele::physics::EosType (which lives in PelePhysics's EOS.H
    // that we deliberately don't include -- see header comment above).
    // For Phase 2 the only EOS we need is GammaLaw; switch to Fuego / SRK
    // in Phase 4 by adding a #if USE_FUEGO_EOS branch here.
    using PeleEOS = pele::physics::eos::GammaLaw;

    /// Per-phase pressure from canonical (alpha rho e)_k.
    /// Phase 2: GammaLaw treats Y as a no-op (single species).  Signature
    /// kept identical to Solver::EOS::EOS::PhasicPressureFromEnergy so the
    /// integrator-side call sites do not change.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static Set::Scalar PhasicPressureFromEnergy(Set::Scalar E_k,
                                                Set::Scalar alpha,
                                                Set::Scalar gam,
                                                Set::Scalar pi_k,
                                                Set::Scalar small = 1.0e-16)
    {
        // Sanity: PelePhysics has no stiffened-gas, so pi_k MUST be zero
        // when this branch is selected for the gas phase.  Liquid phase
        // (pi_k > 0) should still route through Solver::EOS::EOS::Tammann.
        // We silently ignore pi_k here — caller responsible for routing.
        constexpr Set::Scalar div_floor = 1.0e-30;
        Set::Scalar a_safe = std::max(alpha, div_floor);
        Set::Scalar p_k = (gam - 1.0) * (E_k / a_safe);   // pi_k = 0 for gas
        return std::max(p_k, small);
    }

    /// Per-phase internal energy from per-phase pressure.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static Set::Scalar PhasicEnergyFromPressure(Set::Scalar p_k,
                                                Set::Scalar alpha,
                                                Set::Scalar gam,
                                                Set::Scalar /*pi_k unused*/,
                                                Set::Scalar small = 1.0e-16)
    {
        Set::Scalar numer = std::max(p_k, small);
        return alpha * numer / (gam - 1.0);
    }

    /// Per-phase sound speed.  For GammaLaw (single species) this is
    /// numerically identical to the native EOS — but routed through
    /// PelePhysics so the API surface is already in place for Phase 3
    /// (mixture sound speed with multi-species Y_k will need PeleEOS::RTY2Cs).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static Set::Scalar PhasicSoundSpeed(Set::Scalar rho_k,
                                        Set::Scalar p_k,
                                        Set::Scalar gam,
                                        Set::Scalar /*pi_k unused*/,
                                        Set::Scalar small = 1.0e-16)
    {
        constexpr Set::Scalar div_floor = 1.0e-30;
        Set::Scalar pp = std::max(p_k, small);
        return std::sqrt(gam * pp / std::max(rho_k, div_floor));
    }

    // -----------------------------------------------------------------------
    // PHASE-3 STUBS (single-species today, multi-species when wired up).
    // These are the methods the integrator will need once the state vector
    // is extended to carry Y_k.  Left as compile-time stubs for now.
    // -----------------------------------------------------------------------

    /// Mixture pressure from density, internal energy, and mass fractions.
    /// Phase 2: ignores Y, returns ideal-gas p = (gam-1) rho e.
    /// Phase 3: will call pele::physics::eos::EosType::RYE2P(R, Y, e, p).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static Set::Scalar GasPressureFromRhoEnergy(Set::Scalar rho,
                                                Set::Scalar e,
                                                Set::Scalar gam_single = 1.4,
                                                Set::Scalar small = 1.0e-16)
    {
        // Phase 3 TODO: take Y[] and call PeleEOS::RYE2P
        return std::max((gam_single - 1.0) * rho * e, small);
    }

    /// Mixture temperature.  Phase 3 stub.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static Set::Scalar GasTemperatureFromRhoEnergy(Set::Scalar /*rho*/,
                                                   Set::Scalar e,
                                                   Set::Scalar cv_single = 717.86,
                                                   Set::Scalar small = 1.0e-16)
    {
        // Phase 3 TODO: take Y[] and call PeleEOS::EY2T
        return std::max(e / cv_single, small);
    }

    /// Identifier string for logging.
    /// PeleEOS::identifier() returns std::string by value -- so we cache
    /// it in a function-local static to avoid dangling .c_str() pointers.
    static const char* identifier()
    {
        static const std::string id = PeleEOS::identifier();
        return id.c_str();
    }
};

}  // namespace EOS
}  // namespace Solver

#endif // PELE_ENABLED
#endif // SOLVER_EOS_PELEPHYSICSEOS_H
