#ifndef INTEGRATOR_HYDRO2_PELE_H
#define INTEGRATOR_HYDRO2_PELE_H
#include <algorithm>
#include <cmath>
#include "Set/Base.H"
// Hydro2_Pele.H -- thermally perfect species and finite-rate chemistry from a PelePhysics MECHANISM
// (species.eos = pele, chem.backend = pele).  COMPILED ONLY when the build is given a mechanism directory:
//     make ... HYDRO2_MECH=ext/AMReX-Combustion/PelePhysics/Mechanisms/BurkeDryer POSTFIX=2d-g++-<tag> AMREX_TARGET=...
// which defines HYDRO2_PELE_MECH and puts <HYDRO2_MECH>/mechanism.H on the include path.  The default build does not
// see any of this (Thermo::available = false and the two options abort).
//
// WHAT IS USED FROM PELEPHYSICS: the Fuego-generated mechanism.H / mechanism.cpp of the chosen mechanism only --
// molecular weights, NASA-polynomial internal energies and heat capacities (CKUMS, CKCVMS), species names
// (CKSYMS_STR) and the molar production rates (CKWC).  NOT used: PelePhysics' EOS / reactor / transport classes and
// SUNDIALS; the constant-volume reactor below has its own adaptive integrator.
//
// UNITS.  The mechanism is CGS (erg/g, mol/cm^3).  Hydro2 decks that use this file must be SI (kg, m, s, K, Pa, J).
//
// ENERGY CONVENTION (same as the native backend, Hydro2_Chem.H).  The phase energy of the solver stays a SENSIBLE
// energy that is positive and proportional to T at low temperature:
//     e_s,i(T) = e_i(T) - e0_i,   e0_i = e_i(T_lo) - c_v,i(T_lo) T_lo,   T_lo = 200 K
// (e_i = internal energy of species i with its heat of formation; below T_lo and above T_hi = 6000 K the energy is
// continued linearly with the c_v at the bound, so e_s,i = c_v,i(T_lo) T below T_lo).  The chemical energy
// sum_i Y_i e0_i is NOT stored: a reaction step that changes Y at constant volume adds
//     dq = - sum_i dY_i e0_i      to the phase energy and to rho E     (heat released, positive for combustion).
//
// EQUATION OF STATE SEEN BY THE SOLVER.  Hydro2 computes p = (gamma - 1) E_k/alpha_k - gamma pi and
// T = E_k / (m_k c_v) per phase.  For a mixture of thermally perfect gases (pi = 0) with specific sensible energy
// e_s = E_k/m_k the solver is handed, cell by cell,
//     T from e_s = sum_i Y_i e_s,i(T) (Newton),   c_v,eff = e_s / T,   gamma_eff = 1 + R / c_v,eff,  R = sum Y_i R_u/W_i,
// so that p = rho R T and T are EXACT.  What is approximate: the sound speed uses gamma_eff instead of the frozen
// c_p/c_v (a few per cent; it enters the wave-speed estimates and the time step only), and gamma_eff is frozen
// across a Riemann fan.
#ifdef HYDRO2_PELE_MECH
#include "mechanism.H"
#endif
namespace Integrator
{
namespace Pele
{
#ifdef HYDRO2_PELE_MECH
constexpr int NS = NUM_SPECIES;
constexpr bool available = true;
#else
constexpr int NS = 1;
constexpr bool available = false;
#endif
constexpr Set::Scalar T_LO = 200.0, T_HI = 6000.0;

// Species data in SI, filled once (Thermo::Init) and copied by value into the kernels.
struct Thermo
{
    int on = 0;                 // 1 = species.eos = pele
    int ph = 1;                 // phase that carries the mechanism species
    int c0 = 0;                 // first component of that phase in the species field
    // species.pele_sound = frozen (1): the solver is handed a TWO-parameter stiffened-gas fit per cell,
    //     gamma_hat = (p + A + B)/(p + B),   pi_hat = A/gamma_hat - p,   A = rho c_f^2 = gamma_f p,  B = rho e_s,
    // (gamma_f = c_p/c_v at T), which reproduces p, e_s AND the frozen sound speed c_f exactly;
    // c_v,eff = e_s/T keeps the solver's temperature (p + gamma pi)/(rho c_v (gamma - 1)) exact.  0 (default, "effective"): the one-parameter fit above, whose
    // sound speed is 1.3 - 1.7 % above c_f in burnt hydrogen-oxygen-argon.
    int frozen_c = 0;
    Set::Scalar e0[NS] = {}, Ri[NS] = {}, cvlo[NS] = {}, W[NS] = {};   // J/kg, J/(kg K), J/(kg K), kg/mol

    void Init()
    {
#ifdef HYDRO2_PELE_MECH
        Set::Scalar u[NS], c[NS], w[NS];
        CKUMS(T_LO, u); CKCVMS(T_LO, c); CKWT(w);
        for (int i = 0; i < NS; ++i)
        {
            cvlo[i] = c[i] * 1.0e-4;                       // erg/(g K) -> J/(kg K)
            e0[i] = u[i] * 1.0e-4 - cvlo[i] * T_LO;        // erg/g -> J/kg
            W[i] = w[i] * 1.0e-3;                          // g/mol -> kg/mol
            Ri[i] = 8.31446261815324 / W[i];
        }
#endif
    }

    // Sensible energy [J/kg] and c_v [J/(kg K)] of the mixture Y at T
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void EsCv(const Set::Scalar T, const Set::Scalar *Y, Set::Scalar &es, Set::Scalar &cv) const
    {
        es = 0.0; cv = 0.0;
#ifdef HYDRO2_PELE_MECH
        const Set::Scalar Tc = std::min(std::max(T, T_LO), T_HI);
        Set::Scalar u[NS], c[NS];
        CKUMS(Tc, u); CKCVMS(Tc, c);
        for (int i = 0; i < NS; ++i)
        {
            const Set::Scalar ci = c[i] * 1.0e-4;
            es += Y[i] * (u[i] * 1.0e-4 - e0[i] + ci * (T - Tc));
            cv += Y[i] * ci;
        }
#else
        amrex::ignore_unused(T, Y);
#endif
    }
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar R(const Set::Scalar *Y) const
    {
        Set::Scalar r = 0.0;
        for (int i = 0; i < NS; ++i) r += Y[i] * Ri[i];
        return r;
    }
    // Temperature from the specific sensible energy (monotone: Newton, safeguarded by bisection)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar TfromEs(const Set::Scalar es, const Set::Scalar *Y) const
    {
        Set::Scalar cl = 0.0;
        for (int i = 0; i < NS; ++i) cl += Y[i] * cvlo[i];
        if (!(cl > 0.0)) return T_LO;
        if (es <= cl * T_LO) return std::max(es, 0.0) / cl;             // linear branch below T_lo
        Set::Scalar lo = T_LO, hi = 1.0e5, T = std::min(es / cl, 3000.0), e, cv;
        for (int it = 0; it < 60; ++it)
        {
            EsCv(T, Y, e, cv);
            const Set::Scalar f = e - es;
            if (std::abs(f) <= 1.0e-12 * es) break;
            if (f > 0.0) hi = T; else lo = T;
            Set::Scalar Tn = T - f / cv;
            if (!(Tn > lo && Tn < hi)) Tn = 0.5 * (lo + hi);
            if (std::abs(Tn - T) <= 1.0e-13 * T) { T = Tn; break; }
            T = Tn;
        }
        return T;
    }
    // What the solver needs: gamma_eff and c_v,eff from (e_s, Y), or from (p, rho, Y) at start-up
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void GammaCv(const Set::Scalar es, const Set::Scalar *Y, Set::Scalar &gam, Set::Scalar &cv) const
    {
        const Set::Scalar T = TfromEs(es, Y), r = R(Y);
        Set::Scalar cl = 0.0;
        for (int i = 0; i < NS; ++i) cl += Y[i] * cvlo[i];
        cv = (T > 0.0 && es > 0.0) ? es / T : cl;
        gam = 1.0 + r / cv;
    }
    // two-parameter fit (frozen sound speed exact); rho = phase density
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void GammaPiCv(const Set::Scalar es, const Set::Scalar rho, const Set::Scalar *Y, Set::Scalar &gam, Set::Scalar &pinf, Set::Scalar &cv) const
    {
        const Set::Scalar T = TfromEs(es, Y), r = R(Y);
        Set::Scalar e, cvT;
        EsCv(T, Y, e, cvT);
        if (!(T > 0.0) || !(es > 0.0) || !(cvT > 0.0) || !(rho > 0.0)) { GammaCv(es, Y, gam, cv); pinf = 0.0; return; }
        const Set::Scalar pv = r * T, a = (1.0 + r / cvT) * pv, b = es;      // p/rho, c_f^2, e_s
        gam = (pv + a + b) / (pv + b);
        const Set::Scalar piv = a / gam - pv;                                  // pi_hat / rho
        pinf = rho * piv;
        cv = es / T;         // the solver's temperature is (p + gamma pi)/(rho c_v (gamma - 1)) = e_s/c_v
    }
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar EsFromPRho(const Set::Scalar p, const Set::Scalar rho, const Set::Scalar *Y) const
    {
        Set::Scalar es, cv;
        EsCv(p / (rho * R(Y)), Y, es, cv);
        return es;
    }

    // CONSTANT-VOLUME REACTOR over dt: Y (in/out), rho [kg/m^3] fixed, es (in/out) follows dq = -sum dY e0.
    // Adaptive Cash-Karp Runge-Kutta 4(5) on Y; T is recomputed from the energy at every stage.
    // Returns the number of sub-steps (negative = the step limit was reached, the remainder was not integrated).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int React(Set::Scalar *Y, const Set::Scalar rho, Set::Scalar &es, const Set::Scalar dt,
              const Set::Scalar rtol, const Set::Scalar atol, const int max_sub) const
    {
#ifdef HYDRO2_PELE_MECH
        auto rhs = [&](const Set::Scalar *y, const Set::Scalar e, Set::Scalar *dy)
        {
            Set::Scalar yp[NS], C[NS], wd[NS];
            for (int i = 0; i < NS; ++i) yp[i] = std::max(y[i], 0.0);
            const Set::Scalar T = std::min(std::max(TfromEs(e, yp), T_LO), T_HI);
            for (int i = 0; i < NS; ++i) C[i] = rho * 1.0e-3 * yp[i] / (W[i] * 1.0e3);     // mol/cm^3
            CKWC(T, C, wd);                                                              // mol/(cm^3 s)
            for (int i = 0; i < NS; ++i) dy[i] = wd[i] * (W[i] * 1.0e3) / (rho * 1.0e-3);
        };
        auto energy = [&](const Set::Scalar *y, const Set::Scalar *yref, const Set::Scalar eref)
        {
            Set::Scalar e = eref;
            for (int i = 0; i < NS; ++i) e -= (y[i] - yref[i]) * e0[i];
            return e;
        };
        static constexpr Set::Scalar
            b21 = 0.2, b31 = 3.0 / 40, b32 = 9.0 / 40, b41 = 0.3, b42 = -0.9, b43 = 1.2,
            b51 = -11.0 / 54, b52 = 2.5, b53 = -70.0 / 27, b54 = 35.0 / 27,
            b61 = 1631.0 / 55296, b62 = 175.0 / 512, b63 = 575.0 / 13824, b64 = 44275.0 / 110592, b65 = 253.0 / 4096,
            c1 = 37.0 / 378, c3 = 250.0 / 621, c4 = 125.0 / 594, c6 = 512.0 / 1771,
            d1 = c1 - 2825.0 / 27648, d3 = c3 - 18575.0 / 48384, d4 = c4 - 13525.0 / 55296, d5 = -277.0 / 14336, d6 = c6 - 0.25;
        Set::Scalar t = 0.0, h = dt;
        int n = 0;
        Set::Scalar k1[NS], k2[NS], k3[NS], k4[NS], k5[NS], k6[NS], yt[NS], yn[NS];
        while (t < dt)
        {
            if (n >= max_sub) return -n;
            h = std::min(h, dt - t);
            rhs(Y, es, k1);
            for (int i = 0; i < NS; ++i) yt[i] = Y[i] + h * b21 * k1[i];
            rhs(yt, energy(yt, Y, es), k2);
            for (int i = 0; i < NS; ++i) yt[i] = Y[i] + h * (b31 * k1[i] + b32 * k2[i]);
            rhs(yt, energy(yt, Y, es), k3);
            for (int i = 0; i < NS; ++i) yt[i] = Y[i] + h * (b41 * k1[i] + b42 * k2[i] + b43 * k3[i]);
            rhs(yt, energy(yt, Y, es), k4);
            for (int i = 0; i < NS; ++i) yt[i] = Y[i] + h * (b51 * k1[i] + b52 * k2[i] + b53 * k3[i] + b54 * k4[i]);
            rhs(yt, energy(yt, Y, es), k5);
            for (int i = 0; i < NS; ++i) yt[i] = Y[i] + h * (b61 * k1[i] + b62 * k2[i] + b63 * k3[i] + b64 * k4[i] + b65 * k5[i]);
            rhs(yt, energy(yt, Y, es), k6);
            Set::Scalar err = 0.0;
            for (int i = 0; i < NS; ++i)
            {
                yn[i] = Y[i] + h * (c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c6 * k6[i]);
                const Set::Scalar ei = h * (d1 * k1[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] + d6 * k6[i]);
                err = std::max(err, std::abs(ei) / (atol + rtol * std::max(std::abs(Y[i]), std::abs(yn[i]))));
            }
            ++n;
            if (err <= 1.0)
            {
                // accept: clip round-off negatives, keep sum(Y) = 1 on the largest species, energy from the actual change
                Set::Scalar sum = 0.0; int im = 0;
                for (int i = 0; i < NS; ++i) { yn[i] = std::max(yn[i], 0.0); sum += yn[i]; if (yn[i] > yn[im]) im = i; }
                yn[im] += 1.0 - sum;
                es = energy(yn, Y, es);
                for (int i = 0; i < NS; ++i) Y[i] = yn[i];
                t += h;
                h *= std::min(5.0, 0.9 * std::pow(std::max(err, 1.0e-10), -0.2));
            }
            else
                h *= std::max(0.1, 0.9 * std::pow(err, -0.25));
        }
        return n;
#else
        amrex::ignore_unused(Y, rho, es, dt, rtol, atol, max_sub);
        return 0;
#endif
    }
};
}
}
#endif
