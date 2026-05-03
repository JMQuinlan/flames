#pragma once

#include <algorithm>
#include <cmath>

// Thermodynamic mixing utilities for ideal and stiffened gases
// Designed for diffuse-interface compressible flow solvers.

namespace Thermo_Interp
{
    // Adjust this alias if your scalar type differs
    using Scalar = Set::Scalar; // or: using Scalar = double;

    // -------------------------------------------------------------------------
    // Basic helpers
    // -------------------------------------------------------------------------

    inline Scalar clamp01(Scalar x)
    {
        return std::max(Scalar(0.0), std::min(Scalar(1.0), x));
    }

    // Generic "interpolate in transformed space" helper:
    // forward:  physical -> linearly-mixed quantity
    // inverse:  linearly-mixed quantity -> physical
    template <typename Forward, typename Inverse>
    inline Scalar InterpolateThermo(Scalar eta,
                                    Scalar val1,
                                    Scalar val2,
                                    Forward forward,
                                    Inverse inverse)
    {
        eta = clamp01(eta);

        Scalar q1   = forward(val1);
        Scalar q2   = forward(val2);
        Scalar qmix = (Scalar(1.0) - eta) * q1 + eta * q2;

        return inverse(qmix);
    }

    // -------------------------------------------------------------------------
    // 1) Ideal-gas gamma mixing (nonlinear, harmonic in gamma - 1)
    // -------------------------------------------------------------------------

    inline Scalar InterpolateGamma(Scalar eta,
                                   Scalar gamma1,
                                   Scalar gamma2)
    {
        eta = clamp01(eta);

        Scalar A = eta / (gamma2 - Scalar(1.0))
                 + (Scalar(1.0) - eta) / (gamma1 - Scalar(1.0));

        return Scalar(1.0) + Scalar(1.0) / A;
    }

    inline Scalar InterpolateGamma_Generic(Scalar eta,
                                           Scalar gamma1,
                                           Scalar gamma2)
    {
        auto forward = [](Scalar g) { return Scalar(1.0) / (g - Scalar(1.0)); };
        auto inverse = [](Scalar q) { return Scalar(1.0) + Scalar(1.0) / q; };

        return InterpolateThermo(eta, gamma1, gamma2, forward, inverse);
    }

    // -------------------------------------------------------------------------
    // 2) Internal energy mixing (linear)
    // -------------------------------------------------------------------------

    inline Scalar InterpolateInternalEnergy(Scalar eta,
                                            Scalar e1,
                                            Scalar e2)
    {
        eta = clamp01(eta);
        return (Scalar(1.0) - eta) * e1 + eta * e2;
    }

    // -------------------------------------------------------------------------
    // 3) Cv and Cp mixing (usually linear for ideal gases)
    // -------------------------------------------------------------------------

    inline Scalar InterpolateCv(Scalar eta,
                                Scalar Cv1,
                                Scalar Cv2)
    {
        eta = clamp01(eta);
        return (Scalar(1.0) - eta) * Cv1 + eta * Cv2;
    }

    inline Scalar InterpolateCp(Scalar eta,
                                Scalar Cp1,
                                Scalar Cp2)
    {
        eta = clamp01(eta);
        return (Scalar(1.0) - eta) * Cp1 + eta * Cp2;
    }

    // -------------------------------------------------------------------------
    // 4) Temperature mixing via e = Cv T (ideal gas)
    // -------------------------------------------------------------------------

    inline Scalar InterpolateTemperature(Scalar eta,
                                         Scalar T1,
                                         Scalar T2,
                                         Scalar Cv1,
                                         Scalar Cv2)
    {
        eta = clamp01(eta);

        Scalar e_mix = (Scalar(1.0) - eta) * (Cv1 * T1)
                     + eta * (Cv2 * T2);

        Scalar Cv_mix = InterpolateCv(eta, Cv1, Cv2);

        return e_mix / Cv_mix;
    }

    // -------------------------------------------------------------------------
    // 5) Sound speed mixing for ideal gas
    //     c^2 = gamma (gamma - 1) e
    // -------------------------------------------------------------------------

    inline Scalar SoundSpeedFromGammaE(Scalar gamma,
                                       Scalar e)
    {
        return std::sqrt(gamma * (gamma - Scalar(1.0)) * e);
    }

    inline Scalar InterpolateSoundSpeed(Scalar eta,
                                        Scalar gamma1,
                                        Scalar gamma2,
                                        Scalar e1,
                                        Scalar e2)
    {
        eta = clamp01(eta);

        Scalar gamma_mix = InterpolateGamma(eta, gamma1, gamma2);
        Scalar e_mix     = InterpolateInternalEnergy(eta, e1, e2);

        return SoundSpeedFromGammaE(gamma_mix, e_mix);
    }

    // -------------------------------------------------------------------------
    // 6) Stagnation pressure mixing (ideal gas, A/B construction)
    //     A = eta/(g2-1) + (1-eta)/(g1-1)
    //     B = eta*g2*p0_2/(g2-1) + (1-eta)*g1*p0_1/(g1-1)
    //     p0_mix = B / A
    // -------------------------------------------------------------------------

    inline Scalar InterpolateStagnationPressure(Scalar eta,
                                                Scalar gamma1,
                                                Scalar gamma2,
                                                Scalar p0_1,
                                                Scalar p0_2)
    {
        eta = clamp01(eta);

        Scalar A = eta / (gamma2 - Scalar(1.0))
                 + (Scalar(1.0) - eta) / (gamma1 - Scalar(1.0));

        Scalar B = eta * gamma2 * p0_2 / (gamma2 - Scalar(1.0))
                 + (Scalar(1.0) - eta) * gamma1 * p0_1 / (gamma1 - Scalar(1.0));

        return B / A;
    }

    // -------------------------------------------------------------------------
    // 7) Stiffened-gas (Tammann) EOS helpers
    //     p = (gamma - 1) * rho * e - gamma * pi
    // -------------------------------------------------------------------------

    // Mix gamma and pi for a stiffened gas using A/B construction:
    //   A = eta/(g2-1) + (1-eta)/(g1-1)
    //   B = eta*g2*pi2/(g2-1) + (1-eta)*g1*pi1/(g1-1)
    //   gamma_mix = 1 + 1/A
    //   pi_mix    = B / (A + 1)   [NOT B/A — see derivation below]
    //
    // Derivation: for p = (gmix-1)*UE - gmix*pimix to recover p from
    // UE = p*A + B (pressure-equilibrium mixture energy), we need:
    //   (gmix-1)*A = 1  →  gmix = 1+1/A  ✓
    //   pimix = (gmix-1)/gmix * B/A = (1/A)/(1+1/A) * B/A = B/(A+1)
    // Equivalently: at pure phase i, pimix = pi_i (correct single-phase limit).
    inline void InterpolateGammaPi_Stiffened(
        Scalar eta,
        Scalar gamma_eta0, Scalar gamma_eta1,
        Scalar pi_eta0,    Scalar pi_eta1,
        Scalar &gamma_mix,
        Scalar &pi_mix)
    {
        eta = clamp01(eta);

        Scalar A = eta / (gamma_eta1 - Scalar(1.0))
                 + (Scalar(1.0) - eta) / (gamma_eta0 - Scalar(1.0));

        Scalar B = eta * gamma_eta1 * pi_eta1 / (gamma_eta1 - Scalar(1.0))
                 + (Scalar(1.0) - eta) * gamma_eta0 * pi_eta0 / (gamma_eta0 - Scalar(1.0));

        gamma_mix = Scalar(1.0) + Scalar(1.0) / A;
        pi_mix    = B / (A + Scalar(1.0));
    }

    // Stiffened-gas pressure from EOS:
    //   p = (gamma - 1) * rho * e - gamma * pi
    inline Scalar Pressure_Stiffened(
        Scalar rho,
        Scalar e,
        Scalar gamma,
        Scalar pi)
    {
        return (gamma - Scalar(1.0)) * rho * e - gamma * pi;
    }

    // Full stiffened-gas pressure mixing:
    //   - rho, e mixed linearly
    //   - gamma, pi mixed nonlinearly via InterpolateGammaPi_Stiffened
    //   - p_mix from stiffened-gas EOS
    inline Scalar InterpolatePressure_Stiffened(
        Scalar eta,
        Scalar rho_eta0, Scalar rho_eta1,
        Scalar e_eta0,   Scalar e_eta1,
        Scalar gamma_eta0, Scalar gamma_eta1,
        Scalar pi_eta0,    Scalar pi_eta1)
    {
        eta = clamp01(eta);

        Scalar rho_mix = (Scalar(1.0) - eta) * rho_eta0 + eta * rho_eta1;
        Scalar e_mix   = (Scalar(1.0) - eta) * e_eta0   + eta * e_eta1;

        Scalar gamma_mix, pi_mix;
        InterpolateGammaPi_Stiffened(eta, gamma_eta0, gamma_eta1, pi_eta0, pi_eta1,
                                     gamma_mix, pi_mix);

        return Pressure_Stiffened(rho_mix, e_mix, gamma_mix, pi_mix);
    }

    // -------------------------------------------------------------------------
    // 7a) Mass-fraction mixing (BS dual-state closure)
    //
    // The volume-fraction (Allaire/Kapila pressure-equilibrium) blend in
    // Section 7 assumes both phases share a common pressure and that eta is
    // a volume fraction.  In the BS dual-state formulation each phase carries
    // its own (rho, M, E) and we instead blend per-phase intensives by
    // mass fraction:
    //
    //   alpha_0 = 1 - eta,   alpha_1 = eta            (volume fractions)
    //   rho_mix = (1-eta) * rho_0 + eta * rho_1
    //   Y_0     = (1-eta) * rho_0 / rho_mix
    //   Y_1     =     eta * rho_1 / rho_mix
    //
    // For a Y-weighted scalar q_mix = Y_0 * q_0 + Y_1 * q_1 (used for p, u
    // in p_int_method=4 and now for diagnostic mixture p, a, T as well).
    // Convention: subscript "0" = phase at eta=0, "1" = phase at eta=1.
    // -------------------------------------------------------------------------

    inline void Mix_MassFraction_A_B(
        Scalar eta,
        Scalar rho0, Scalar rho1,
        Scalar &Y0, Scalar &Y1, Scalar &rho_mix)
    {
        eta = clamp01(eta);
        Scalar a0   = (Scalar(1.0) - eta) * rho0;
        Scalar a1   = eta * rho1;
        Scalar asum = a0 + a1;
        if (asum > Scalar(1e-30))
        {
            Y0      = a0 / asum;
            Y1      = a1 / asum;
            rho_mix = asum;
        }
        else
        {
            Y0      = Scalar(0.5);
            Y1      = Scalar(0.5);
            rho_mix = Scalar(1e-30);
        }
    }

    // Y-weighted (gamma, pi) closure — mass-fraction analogue of the
    // Allaire A/B blend in Section 7:
    //   1/(gamma_Y - 1) = Y_0/(g_0-1) + Y_1/(g_1-1)
    //   pi_Y            = (Y_0 g_0 pi_0/(g_0-1) + Y_1 g_1 pi_1/(g_1-1))
    //                   / ( Y_0/(g_0-1) + Y_1/(g_1-1) + 1 )
    // Reduces exactly to (g_i, pi_i) at the pure-phase limit Y_i = 1.
    // Note: this is a closure for plot/Riemann diagnostics; the BS source
    // and per-phase EOS in Section 7b remain the load-bearing path.
    inline void MixGammaPi_MassFraction(
        Scalar Y0, Scalar Y1,
        Scalar gamma0, Scalar gamma1,
        Scalar pi0,    Scalar pi1,
        Scalar &gamma_Y,
        Scalar &pi_Y)
    {
        Scalar A = Y0 / (gamma0 - Scalar(1.0))
                 + Y1 / (gamma1 - Scalar(1.0));

        Scalar B = Y0 * gamma0 * pi0 / (gamma0 - Scalar(1.0))
                 + Y1 * gamma1 * pi1 / (gamma1 - Scalar(1.0));

        gamma_Y = Scalar(1.0) + Scalar(1.0) / A;
        pi_Y    = B / (A + Scalar(1.0));
    }

    // -------------------------------------------------------------------------
    // 7b) Per-phase single-phase EOS inversions on (rho, UE)
    //     UE = rho * e (volumetric internal energy)
    //     These are used in the dual-state BS formulation where each phase
    //     carries its own (rho, M, E) and is closed by its own single-phase
    //     EOS — no Allaire/Kapila mixing.
    // -------------------------------------------------------------------------

    // Calorically perfect gas: p = (gamma - 1) * UE
    inline Scalar Pressure_CPG(Scalar UE, Scalar gamma)
    {
        return (gamma - Scalar(1.0)) * UE;
    }

    // Stiffened gas from volumetric UE: p = (gamma-1)*UE - gamma*pi
    inline Scalar Pressure_SG_UE(Scalar UE, Scalar gamma, Scalar pi)
    {
        return (gamma - Scalar(1.0)) * UE - gamma * pi;
    }

    // Sound speed for CPG from (rho, p): c^2 = gamma * p / rho
    inline Scalar SoundSpeed_CPG(Scalar rho, Scalar p, Scalar gamma)
    {
        Scalar c2 = gamma * p / rho;
        return c2 > Scalar(0.0) ? std::sqrt(c2) : Scalar(0.0);
    }

    // Sound speed for stiffened gas: c^2 = gamma * (p + pi) / rho
    inline Scalar SoundSpeed_SG(Scalar rho, Scalar p, Scalar gamma, Scalar pi)
    {
        Scalar c2 = gamma * (p + pi) / rho;
        return c2 > Scalar(0.0) ? std::sqrt(c2) : Scalar(0.0);
    }

    // Temperature from specific internal energy and Cv (ideal / SG share form)
    inline Scalar Temperature_FromE(Scalar e, Scalar Cv)
    {
        return (Cv > Scalar(0.0)) ? (e / Cv) : Scalar(0.0);
    }

    // -------------------------------------------------------------------------
    // 8) Generic interface interpolation (linear vs nonlinear gamma)
    // -------------------------------------------------------------------------

    inline Scalar InterpolateInterfaceLinear(Scalar eta,
                                             Scalar val1,
                                             Scalar val2)
    {
        eta = clamp01(eta);
        return (Scalar(1.0) - eta) * val1 + eta * val2;
    }

    inline Scalar InterpolateInterfaceGammaNonlinear(Scalar eta,
                                                     Scalar gamma1,
                                                     Scalar gamma2)
    {
        return InterpolateGamma(eta, gamma1, gamma2);
    }
}
