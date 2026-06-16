// =============================================================================
// pele_hello.cc  -- "Hello PelePhysics" smoke test for Phase 1+2 scaffolding.
//
// Builds as a standalone executable `bin/pele_hello-2d-g++` (the Makefile
// auto-discovers any *.cc under src/).  When PELE_ENABLED is set in the
// build, this exercises:
//   1. PelePhysics headers actually compile against our build flags
//   2. The Solver::EOS::PelePhysicsEOS shim can be instantiated and called
//   3. A round-trip pressure <-> energy at standard conditions matches the
//      ideal-gas analytical answer to machine precision
//
// When PELE_ENABLED is NOT set, the binary prints a friendly opt-in message.
//
// This is the lightest possible verification that Phase 1 wiring is correct
// before tackling Phase 2's larger integration into Hydro2.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include "Util/Util.H"

#ifdef PELE_ENABLED
#include "Solver/EOS/PelePhysicsEOS.H"
#endif

int main(int argc, char* argv[])
{
    Util::Initialize(argc, argv);

#ifndef PELE_ENABLED
    Util::Message(INFO, "pele_hello: PELE_ENABLED is NOT set in this build.");
    Util::Message(INFO,
        "To enable: run scripts/build_pele_deps.sh, then set");
    Util::Message(INFO,
        "  export SUNDIALS_TARGET=$(pwd)/ext/sundials/install");
    Util::Message(INFO,
        "  export PELE_PHYSICS_HOME=$(pwd)/ext/pelephysics");
    Util::Message(INFO,
        "  export PELE_ENABLED=1");
    Util::Message(INFO, "Then rebuild this target.");
    Util::Finalize();
    return 0;
#else
    using EOS = Solver::EOS::PelePhysicsEOS;

    Util::Message(INFO, "pele_hello: PELE_ENABLED build, EOS = ", EOS::identifier());

    // Air-like reference state at STP-ish conditions.
    constexpr Set::Scalar rho   = 1.225;       // [kg/m^3]
    constexpr Set::Scalar T     = 300.0;       // [K]
    constexpr Set::Scalar gam   = 1.4;
    constexpr Set::Scalar cv    = 717.86;      // [J/(kg K)] (air, ideal)
    constexpr Set::Scalar pi_k  = 0.0;         // ideal gas, no Tammann

    // Compute internal energy at this T (e = c_v T for ideal gas)
    Set::Scalar e   = cv * T;                  // ~ 215358 J/kg
    Set::Scalar rho_e = rho * e;               // [J/m^3]

    // Treat as a pure-gas cell: alpha = 1.0, E_k = rho_e (full mass in this phase).
    Set::Scalar p = EOS::PhasicPressureFromEnergy(rho_e, 1.0, gam, pi_k);

    Set::Scalar p_expected = (gam - 1.0) * rho_e;        // ~ 101325 Pa
    Set::Scalar rel_err    = std::abs(p - p_expected) / p_expected;

    Util::Message(INFO, "  rho             = ", rho,  " kg/m^3");
    Util::Message(INFO, "  T               = ", T,    " K");
    Util::Message(INFO, "  e (= cv*T)      = ", e,    " J/kg");
    Util::Message(INFO, "  p (shim)        = ", p,    " Pa");
    Util::Message(INFO, "  p (analytical)  = ", p_expected, " Pa");
    Util::Message(INFO, "  relative error  = ", rel_err);

    // Round-trip: E(p) should recover rho_e.
    Set::Scalar rho_e_rt = EOS::PhasicEnergyFromPressure(p, 1.0, gam, pi_k);
    Set::Scalar rt_err   = std::abs(rho_e_rt - rho_e) / rho_e;
    Util::Message(INFO, "  rho*e round-trip err = ", rt_err);

    // Sound speed sanity.  For ideal-gas air at 300 K (NOT 15 C / 288 K):
    //   c = sqrt(gamma R_specific T) = sqrt(1.4 * 287 * 300) = 347.2 m/s
    // (At ~288 K it's the more commonly-quoted ~340 m/s; I had that value
    // in the original assertion, which was wrong for THIS test state.)
    Set::Scalar a = EOS::PhasicSoundSpeed(rho, p, gam, pi_k);
    const Set::Scalar a_expected = std::sqrt(gam * p / rho);
    Util::Message(INFO, "  sound speed     = ", a, " m/s   (expected ", a_expected, ")");

    const bool ok = (rel_err < 1.0e-12) && (rt_err < 1.0e-12)
                 && std::abs(a - a_expected) < 1.0e-6;
    Util::Message(INFO, "RESULT: ", ok ? "PASS" : "FAIL");

    Util::Finalize();
    return ok ? 0 : 1;
#endif
}
