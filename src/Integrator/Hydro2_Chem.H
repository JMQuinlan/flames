#ifndef INTEGRATOR_HYDRO2_CHEM_H
#define INTEGRATOR_HYDRO2_CHEM_H
#include <algorithm>
#include <cmath>
#include <string>
#include "Set/Base.H"
namespace Integrator {
// Hydro2_Chem.H -- CHEMISTRY on the species of Hydro2, default OFF (chem.on = 0).  Needs species.on = 1.
// STATUS 2026-10-06: VERIFIED on tests/FlowChemistry -- constant-volume reactor (Y = exp(-A t) to 2e-12, thermal
// explosion to 1e-6) and Chapman-Jouguet detonation (front speed +0.3 %, von Neumann peak -1.1 %, energy -6e-6);
// use Riemann_Solver.type = hlle for the detonation (HLLC shows a transverse instability on strong aligned shocks).
// THIS IS THE "NATIVE" BACKEND OF THE DESIGN (docs/design/Hydro2_MultiSpecies_Chemistry_Design.md, stage 4): one
// irreversible one-step Arrhenius reaction inside one phase,
//        reactant -> product + Q,      dY_r/dt = -A Y_r exp(-T_a / T),
// Y_r = mass fraction of the reactant IN ITS PHASE, T = temperature of that phase, Q = heat released per unit mass of
// reactant consumed.  It exists so that the transport / energy / splitting machinery can be verified against closed
// forms (constant-volume reactor, Chapman-Jouguet speed, ZND structure) before a library (PelePhysics, Mutation++)
// supplies real mechanisms through the same hook.
//
// ENERGY CONVENTION: the phase energies stay SENSIBLE (as in the phase-change module); the heat of reaction is booked
// when the reaction happens:  E_k += Q dm,  rho E += Q dm.  Reactant and product share the equation of state of their
// phase (species thermodynamics is stage 2 of the design).
//
// INTEGRATION: operator split (Lie), once per step after the flow update, at constant volume and constant phase mass:
//        cv dT/dt = -Q dY_r/dt   ->   T = T_0 + Q (Y_0 - Y)/cv      (exact)
// so one scalar equation dY/dt = -A Y exp(-T_a/T(Y)) remains; it is integrated with classical RK4 in sub-steps that
// change Y by at most a fraction `dY_max` of itself or T by `dT_max` of itself (thermal runaway).
//
// INPUT   chem.on = 1   chem.reactant = <species name>   chem.product = <species name of the SAME phase>
//         chem.A  (1/s)   chem.Ta (activation temperature E_a/R, in the temperature unit of the run)   chem.Q (J/kg)
//         chem.T_min (default 0): no reaction below this temperature (suppresses numerical pre-ignition)
// OUTPUT  plot field chem_heat = heat release rate per unit volume of the last step.
class Chem {
public:
    int on = 0;
    std::string reactant, product;
    int ir = -1, ip = -1, phase = -1;        // component of the reactant / product in the species field; their phase
    Set::Scalar A = 0.0, Ta = 0.0, Q = 0.0, T_min = 0.0;
    Set::Scalar dY_max = 0.05, dT_max = 0.02;
    // [2026-11-18] GLOBAL FUEL + OXIDIZER STEP (chem.oxidizer, default none = the one-reactant law above):
    //     F + s O -> (1 + s) P,     dY_F/dt = -A rho^(a + b - 1) Y_F^a Y_O^b exp(-T_a / T),     Y_O = Y_O,0 - s (Y_F,0 - Y_F)
    // (the Westbrook & Dryer (1981) global form in mass fractions; their molar-concentration constants convert into A).
    // chem.s = stoichiometric oxidizer-to-fuel MASS ratio, chem.order_fuel = a (1), chem.order_ox = b (1).  Q per kg fuel.
    std::string oxidizer;
    int iox = -1;
    Set::Scalar s_ox = 0.0, a_F = 1.0, b_O = 1.0;
    // chem.alpha_min (default 0): no reaction where the reacting phase holds less than this volume fraction (its
    // temperature in a trace phase -- the gas inside the liquid side of an interface -- is not reliable)
    Set::Scalar alpha_min = 0.0;
    int max_sub = 20000;
    int pele = 0;                            // chem.backend = pele: mechanism of the build (Hydro2_Pele.H)
    Set::Scalar rtol = 1.0e-6, atol = 1.0e-12;   // error control of the pele reactor (on the mass fractions)
    Set::Field<Set::Scalar> heat_mf;

    // The rate law and its integrator, as a plain copyable object for the kernels.
    struct Rate
    {
        Set::Scalar A, Ta, Q, T_min, dY_max, dT_max; int max_sub;
        // Advance the reactant mass fraction Y over dt at constant volume from temperature T0; cv of the phase.
        // Returns the new Y (T follows from the exact relation in the header).
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        Set::Scalar React(const Set::Scalar Y0, const Set::Scalar T0, const Set::Scalar cv, const Set::Scalar dt) const
        {
            if (!(Y0 > 0.0) || !(T0 > T_min)) return Y0;
            const Set::Scalar A_ = A, Ta_ = Ta, Qc = Q / cv;
            auto f = [=](Set::Scalar Y) { const Set::Scalar T = T0 + Qc * (Y0 - Y); return -A_ * Y * std::exp(-Ta_ / T); };
            Set::Scalar Y = Y0, t = 0.0;
            for (int n = 0; n < max_sub && t < dt && Y > 1.0e-300; ++n)
            {
                const Set::Scalar T = T0 + Qc * (Y0 - Y), r = -f(Y);                 // r = -dY/dt > 0
                if (!(r > 0.0)) break;
                // sub-step: Y changes by <= dY_max Y and T by <= dT_max T
                Set::Scalar h = std::min(dY_max * Y / r, dT_max * T / std::max(std::abs(Qc) * r, 1.0e-300));
                h = std::min(h, dt - t);
                const Set::Scalar k1 = f(Y), k2 = f(std::max(Y + 0.5 * h * k1, 0.0)), k3 = f(std::max(Y + 0.5 * h * k2, 0.0)), k4 = f(std::max(Y + h * k3, 0.0));
                Y = std::max(Y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 0.0);
                t += h;
            }
            return Y;
        }
    };
    Rate rate() const { return Rate{A, Ta, Q, T_min, dY_max, dT_max, max_sub}; }

    // The global fuel + oxidizer law (chem.oxidizer).  Same sub-stepping and RK4 as Rate::React.
    struct Rate2
    {
        Set::Scalar A, Ta, Q, T_min, dY_max, dT_max; int max_sub; Set::Scalar s, a, b;
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        Set::Scalar React(const Set::Scalar Y0, const Set::Scalar Yo0, const Set::Scalar rho, const Set::Scalar T0,
                          const Set::Scalar cv, const Set::Scalar dt) const
        {
            if (!(Y0 > 0.0) || !(Yo0 > 0.0) || !(T0 > T_min)) return Y0;
            const Set::Scalar Qc = Q / cv, Ar = A * std::pow(rho, a + b - 1.0), s_ = s, a_ = a, b_ = b, Ta_ = Ta;
            const Set::Scalar Ymin = std::max(Y0 - Yo0 / s_, 0.0);        // the oxidizer runs out at Y = Ymin
            auto f = [=](Set::Scalar Y) {
                const Set::Scalar T = T0 + Qc * (Y0 - Y), Yo = std::max(Yo0 - s_ * (Y0 - Y), 0.0);
                return -Ar * std::pow(std::max(Y, 0.0), a_) * std::pow(Yo, b_) * std::exp(-Ta_ / T); };
            Set::Scalar Y = Y0, t = 0.0;
            for (int n = 0; n < max_sub && t < dt && Y > Ymin + 1.0e-300; ++n)
            {
                const Set::Scalar T = T0 + Qc * (Y0 - Y), r = -f(Y);
                if (!(r > 0.0)) break;
                const Set::Scalar lim = std::min(Y, std::max(Yo0 - s_ * (Y0 - Y), 0.0) / s_);   // whichever reactant limits
                Set::Scalar h = std::min(dY_max * lim / r, dT_max * T / std::max(std::abs(Qc) * r, 1.0e-300));
                h = std::min(h, dt - t);
                const Set::Scalar k1 = f(Y), k2 = f(std::max(Y + 0.5 * h * k1, Ymin)), k3 = f(std::max(Y + 0.5 * h * k2, Ymin)), k4 = f(std::max(Y + h * k3, Ymin));
                Y = std::max(Y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, Ymin);
                t += h;
            }
            return Y;
        }
    };
    Rate2 rate2() const { return Rate2{A, Ta, Q, T_min, dY_max, dT_max, max_sub, s_ox, a_F, b_O}; }
};
}
#endif
