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
    //   pi_mix    = B / A
    inline void InterpolateGammaPi_Stiffened(
        Scalar eta,
        Scalar gamma1, Scalar gamma2,
        Scalar pi1,    Scalar pi2,
        Scalar &gamma_mix,
        Scalar &pi_mix)
    {
        eta = clamp01(eta);

        Scalar A = eta / (gamma2 - Scalar(1.0))
                 + (Scalar(1.0) - eta) / (gamma1 - Scalar(1.0));

        Scalar B = eta * gamma2 * pi2 / (gamma2 - Scalar(1.0))
                 + (Scalar(1.0) - eta) * gamma1 * pi1 / (gamma1 - Scalar(1.0));

        gamma_mix = Scalar(1.0) + Scalar(1.0) / A;
        pi_mix    = B / A;
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
        Scalar rho1, Scalar rho2,
        Scalar e1,   Scalar e2,
        Scalar gamma1, Scalar gamma2,
        Scalar pi1,    Scalar pi2)
    {
        eta = clamp01(eta);

        Scalar rho_mix = (Scalar(1.0) - eta) * rho1 + eta * rho2;
        Scalar e_mix   = (Scalar(1.0) - eta) * e1   + eta * e2;

        Scalar gamma_mix, pi_mix;
        InterpolateGammaPi_Stiffened(eta, gamma1, gamma2, pi1, pi2,
                                     gamma_mix, pi_mix);

        return Pressure_Stiffened(rho_mix, e_mix, gamma_mix, pi_mix);
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
