// This implements a modified Riemann HLLC solver for weakly incompressible flow using the Tammann EOS
// with 5th order WENO reconstruction for high-order accuracy.
//
// HLLC solver uses the fastest left and right wave speeds, along with the middle contact wave.
// Restores some accuracy when compared to HLLE with a slightly increased computational time, while providing same robustness.
//
// Source: V.A. Titarev, E.F. Toro, Finite-volume WENO schemes for three-dimensional conservation laws,
// Journal of Computational Physics, Volume 201, Issue 1, 2004, Pages 238-260, ISSN 0021-9991,
// https://doi.org/10.1016/j.jcp.2004.05.015.
//          URL: https://www.sciencedirect.com/science/article/pii/S0021999104002281
//
// Source: M. J. Del Razo and R. J. LeVeque, "Numerical Methods for Interface Coupling of Compressible
// and Almost Incompressible Media," SIAM Journal on Scientific Computing, vol. 39, no. 3, pp. B486-B507,
// Jan. 2017, doi: 10.1137/16m1067834.
//          URL: https://arxiv.org/abs/1604.08544
//

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_WENO5_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_WENO5_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
//#include "Util/Util.H"
#include <cmath>
#include <vector>

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// WENO5 reconstruction class
class WENO5
{
public:
    WENO5() : eps_(1.0e-6) {}

    // WENO5 coefficients
    static constexpr double C13 = 1.0 / 3.0;
    static constexpr double C23 = 2.0 / 3.0;
    static constexpr double C16 = 1.0 / 6.0;
    static constexpr double C56 = 5.0 / 6.0;
    static constexpr double C10 = 1.0 / 10.0;
    static constexpr double C310 = 3.0 / 10.0;
    static constexpr double C610 = 6.0 / 10.0;

    // Reconstruct a single component using WENO5
    double reconstructLeft(const double *v, int i) const
    {
        // Compute candidate stencils for left state (i+1/2, right side of cell i)
        double s1 = C13 * v[i - 2] - C23 * v[i - 1] + C16 * v[i];
        double s2 = -C16 * v[i - 1] + C56 * v[i] + C13 * v[i + 1];
        double s3 = C13 * v[i] + C23 * v[i + 1] - C16 * v[i + 2];

        // Compute smoothness indicators
        double beta1 = (13.0 / 12.0) * pow(v[i - 2] - 2.0 * v[i - 1] + v[i], 2) + (1.0 / 4.0) * pow(v[i - 2] - 4.0 * v[i - 1] + 3.0 * v[i], 2);
        double beta2 = (13.0 / 12.0) * pow(v[i - 1] - 2.0 * v[i] + v[i + 1], 2) + (1.0 / 4.0) * pow(v[i - 1] - v[i + 1], 2);
        double beta3 = (13.0 / 12.0) * pow(v[i] - 2.0 * v[i + 1] + v[i + 2], 2) + (1.0 / 4.0) * pow(3.0 * v[i] - 4.0 * v[i + 1] + v[i + 2], 2);

        // Compute nonlinear weights
        double w1t = C10 / pow(eps_ + beta1, 2);
        double w2t = C610 / pow(eps_ + beta2, 2);
        double w3t = C310 / pow(eps_ + beta3, 2);
        double sum = w1t + w2t + w3t;

        return (w1t * s1 + w2t * s2 + w3t * s3) / sum;
    }

    double reconstructRight(const double *v, int i) const
    {
        // Compute candidate stencils for right state (i-1/2, left side of cell i)
        double s1 = C13 * v[i + 2] - C23 * v[i + 1] + C16 * v[i];
        double s2 = -C16 * v[i + 1] + C56 * v[i] + C13 * v[i - 1];
        double s3 = C13 * v[i] + C23 * v[i - 1] - C16 * v[i - 2];

        // Compute smoothness indicators
        double beta1 = (13.0 / 12.0) * pow(v[i + 2] - 2.0 * v[i + 1] + v[i], 2) + (1.0 / 4.0) * pow(v[i + 2] - 4.0 * v[i + 1] + 3.0 * v[i], 2);
        double beta2 = (13.0 / 12.0) * pow(v[i + 1] - 2.0 * v[i] + v[i - 1], 2) + (1.0 / 4.0) * pow(v[i + 1] - v[i - 1], 2);
        double beta3 = (13.0 / 12.0) * pow(v[i] - 2.0 * v[i - 1] + v[i - 2], 2) + (1.0 / 4.0) * pow(3.0 * v[i] - 4.0 * v[i - 1] + v[i - 2], 2);

        // Compute nonlinear weights
        double w1t = C10 / pow(eps_ + beta1, 2);
        double w2t = C610 / pow(eps_ + beta2, 2);
        double w3t = C310 / pow(eps_ + beta3, 2);
        double sum = w1t + w2t + w3t;

        return (w1t * s1 + w2t * s2 + w3t * s3) / sum;
    }

    // Reconstruct states at cell interfaces (component-wise)
    void reconstructStates(const std::vector<State> &cellAverages,
                           std::vector<State> &leftStates,
                           std::vector<State> &rightStates) const
    {
        const int n = cellAverages.size();
        leftStates.resize(n + 1);
        rightStates.resize(n + 1);

        // Extract components for reconstruction
        std::vector<double> rho(n), Mn(n), Mt(n), E(n);
        for (int i = 0; i < n; i++)
        {
            rho[i] = cellAverages[i].rho;
            Mn[i] = cellAverages[i].M_normal;
            Mt[i] = cellAverages[i].M_tangent;
            E[i] = cellAverages[i].E;
        }

        // Add ghost cells (simple extrapolation)
        rho.insert(rho.begin(), 2, rho[0]);
        rho.insert(rho.end(), 2, rho[n - 1]);
        Mn.insert(Mn.begin(), 2, Mn[0]);
        Mn.insert(Mn.end(), 2, Mn[n - 1]);
        Mt.insert(Mt.begin(), 2, Mt[0]);
        Mt.insert(Mt.end(), 2, Mt[n - 1]);
        E.insert(E.begin(), 2, E[0]);
        E.insert(E.end(), 2, E[n - 1]);

        // Perform reconstruction for each interface
        for (int i = 0; i <= n; i++)
        {
            int idx = i + 2;
            if (i == 0)
            {
                leftStates[i] = cellAverages[0];
                rightStates[i] = cellAverages[0];
            }
            else if (i == n)
            {
                leftStates[i] = cellAverages[n - 1];
                rightStates[i] = cellAverages[n - 1];
            }
            else
            {
                rightStates[i].rho = reconstructRight(rho.data(), idx);
                rightStates[i].M_normal = reconstructRight(Mn.data(), idx);
                rightStates[i].M_tangent = reconstructRight(Mt.data(), idx);
                rightStates[i].E = reconstructRight(E.data(), idx);

                leftStates[i].rho = reconstructLeft(rho.data(), idx - 1);
                leftStates[i].M_normal = reconstructLeft(Mn.data(), idx - 1);
                leftStates[i].M_tangent = reconstructLeft(Mt.data(), idx - 1);
                leftStates[i].E = reconstructLeft(E.data(), idx - 1);
            }
        }
    }

    // Characteristic-wise reconstruction (more accurate for systems of equations)
    void reconstructCharacteristicStates(const std::vector<State> &cellAverages,
                                         std::vector<State> &leftStates,
                                         std::vector<State> &rightStates,
                                         Set::Scalar small,
                                         int Spec_Vol = 1) const
    {
        const int n = cellAverages.size();
        leftStates.resize(n + 1);
        rightStates.resize(n + 1);

        // Add ghost cells
        std::vector<State> extendedCells = cellAverages;
        extendedCells.insert(extendedCells.begin(), 2, cellAverages[0]);
        extendedCells.insert(extendedCells.end(), 2, cellAverages[n - 1]);

        // For each interface
        for (int i = 0; i <= n; i++)
        {
            int idx = i + 2;
            if (i == 0 || i == n)
            {
                if (i == 0)
                {
                    leftStates[i] = cellAverages[0];
                    rightStates[i] = cellAverages[0];
                }
                else
                {
                    leftStates[i] = cellAverages[n - 1];
                    rightStates[i] = cellAverages[n - 1];
                }
            }
            else
            {
                // Compute Roe average at interface
                Set::Scalar rho_L = extendedCells[idx - 1].rho;
                Set::Scalar rho_R = extendedCells[idx].rho;
                Set::Scalar gamma_L = extendedCells[idx - 1].gam;
                Set::Scalar gamma_R = extendedCells[idx].gam;
                Set::Scalar p0_L = extendedCells[idx - 1].p0;
                Set::Scalar p0_R = extendedCells[idx].p0;

                Set::Scalar sqrt_rho_L = std::sqrt(rho_L);
                Set::Scalar sqrt_rho_R = std::sqrt(rho_R);

                Set::Scalar u_L = extendedCells[idx - 1].M_normal / (rho_L + small);
                Set::Scalar u_R = extendedCells[idx].M_normal / (rho_R + small);
                Set::Scalar v_L = extendedCells[idx - 1].M_tangent / (rho_L + small);
                Set::Scalar v_R = extendedCells[idx].M_tangent / (rho_R + small);

                Set::Scalar E_L = extendedCells[idx - 1].E;
                Set::Scalar E_R = extendedCells[idx].E;

                // Calculate primitives based on Spec_Vol
                Set::Scalar ke_L, ke_R, p_L, p_R, H_L, H_R;
                if (Spec_Vol == 1)
                {
                    ke_L = 0.5 * (u_L * u_L + v_L * v_L) * rho_L;
                    ke_R = 0.5 * (u_R * u_R + v_R * v_R) * rho_R;
                    p_L = (gamma_L - 1.0) * (E_L - ke_L) - gamma_L * p0_L;
                    p_R = (gamma_R - 1.0) * (E_R - ke_R) - gamma_R * p0_R;
                    H_L = (E_L + p_L) / rho_L;
                    H_R = (E_R + p_R) / rho_R;
                }
                else
                {
                    ke_L = 0.5 * (u_L * u_L + v_L * v_L);
                    ke_R = 0.5 * (u_R * u_R + v_R * v_R);
                    p_L = (gamma_L - 1.0) * rho_L * (E_L - ke_L) - gamma_L * p0_L;
                    p_R = (gamma_R - 1.0) * rho_R * (E_R - ke_R) - gamma_R * p0_R;
                    H_L = E_L + p_L / rho_L;
                    H_R = E_R + p_R / rho_R;
                }

                // Roe averages
                Set::Scalar u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R + small);
                Set::Scalar v_roe = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) / (sqrt_rho_L + sqrt_rho_R + small);
                Set::Scalar H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R + small);

                Set::Scalar p_roe = (sqrt_rho_L * p_L + sqrt_rho_R * p_R) / (sqrt_rho_L + sqrt_rho_R + small);
                Set::Scalar p0_roe = (sqrt_rho_L * p0_L + sqrt_rho_R * p0_R) / (sqrt_rho_L + sqrt_rho_R + small);
                Set::Scalar gamma_roe = (sqrt_rho_L * gamma_L + sqrt_rho_R * gamma_R) / (sqrt_rho_L + sqrt_rho_R + small);

                Set::Scalar a_roe = std::sqrt(gamma_roe * (p_roe + p0_roe) / (sqrt_rho_L * sqrt_rho_R + small));

                // Right eigenvectors matrix (R)
                double R[4][4] = {
                    { 1.0, 1.0, 1.0, 0.0 },
                    { u_roe - a_roe, u_roe, u_roe + a_roe, 0.0 },
                    { v_roe, v_roe, v_roe, 1.0 },
                    { H_roe - u_roe * a_roe, 0.5 * (u_roe * u_roe + v_roe * v_roe), H_roe + u_roe * a_roe, v_roe }
                };

                // Left eigenvectors matrix (L = R^-1)
                double b1 = (gamma_roe - 1.0) / (a_roe * a_roe + small);
                double b2 = 0.5 * (u_roe * u_roe + v_roe * v_roe);

                double L[4][4] = {
                    { 0.5 * (b1 * b2 + u_roe / a_roe), -0.5 * (b1 * u_roe + 1.0 / a_roe), -0.5 * b1 * v_roe, 0.5 * b1 },
                    { 1.0 - b1 * b2, b1 * u_roe, b1 * v_roe, -b1 },
                    { 0.5 * (b1 * b2 - u_roe / a_roe), -0.5 * (b1 * u_roe - 1.0 / a_roe), -0.5 * b1 * v_roe, 0.5 * b1 },
                    { -v_roe, 0.0, 1.0, 0.0 }
                };

                // Project to characteristic variables
                std::vector<std::vector<double> > charVars(4, std::vector<double>(5));
                for (int j = idx - 3; j <= idx + 1; j++)
                {
                    double U[4] = {
                        extendedCells[j].rho,
                        extendedCells[j].M_normal,
                        extendedCells[j].M_tangent,
                        extendedCells[j].E
                    };

                    for (int k = 0; k < 4; k++)
                    {
                        charVars[k][j - (idx - 3)] = 0.0;
                        for (int m = 0; m < 4; m++)
                        {
                            charVars[k][j - (idx - 3)] += L[k][m] * U[m];
                        }
                    }
                }

                // Apply WENO5 to each characteristic field
                double leftChar[4], rightChar[4];
                for (int k = 0; k < 4; k++)
                {
                    rightChar[k] = reconstructRight(charVars[k].data(), 3);
                    leftChar[k] = reconstructLeft(charVars[k].data(), 2);
                }

                // Project back to physical space
                double leftU[4] = { 0.0 }, rightU[4] = { 0.0 };
                for (int k = 0; k < 4; k++)
                {
                    for (int m = 0; m < 4; m++)
                    {
                        leftU[k] += R[k][m] * leftChar[m];
                        rightU[k] += R[k][m] * rightChar[m];
                    }
                }

                // Convert back to State
                leftStates[i].rho = std::max(small, leftU[0]);
                leftStates[i].M_normal = leftU[1];
                leftStates[i].M_tangent = leftU[2];
                leftStates[i].E = leftU[3];
                leftStates[i].gam = gamma_L;
                leftStates[i].p0 = p0_L;

                rightStates[i].rho = std::max(small, rightU[0]);
                rightStates[i].M_normal = rightU[1];
                rightStates[i].M_tangent = rightU[2];
                rightStates[i].E = rightU[3];
                rightStates[i].gam = gamma_R;
                rightStates[i].p0 = p0_R;
            }
        }
    }

private:
    double eps_; // Small parameter to avoid division by zero
};

/// HLLC Riemann Solver with WENO5 reconstruction
class HLLC_WENO5 : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc_weno5";

    HLLC_WENO5(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    HLLC_WENO5(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC_WENO5()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    bool use_characteristic_decomp = true;

    static void Parse(HLLC_WENO5 &value, IO::ParmParse &pp)
    {
        pp.query_default("use_characteristic_decomp", value.use_characteristic_decomp, true);
    }

    // Solve for a single interface with given left and right states
    virtual Flux Solve(State lo, State hi, Set::Scalar small, int Spec_Vol = 1) override
    {
        Set::Scalar rho_L = lo.rho, rho_R = hi.rho;
        Set::Scalar Mn_L = lo.M_normal, Mn_R = hi.M_normal;
        Set::Scalar Mt_L = lo.M_tangent, Mt_R = hi.M_tangent;
        Set::Scalar E_L = lo.E, E_R = hi.E;
        Set::Scalar gamma_L = lo.gam, gamma_R = hi.gam;
        Set::Scalar p0_L = lo.p0, p0_R = hi.p0;
        Set::Scalar T_L = lo.T, T_R = hi.T;

        // Ensure no negative densities
        rho_L = std::max(small, rho_L);
        rho_R = std::max(small, rho_R);

        // Fluid primitives
        Set::Scalar ke_L, ke_R;
        Set::Scalar ue_L, ue_R;
        Set::Scalar p_L, p_R;
        Set::Scalar h_TL, h_TR;

        if (Spec_Vol == 1)
        {
            // With Respect to Specific Volume (energies are per unit volume)
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L + small);
            ue_L = E_L - ke_L;
            p_L = (gamma_L - 1.0) * ue_L - gamma_L * p0_L;
            h_TL = (ke_L + ue_L + p_L);

            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R + small);
            ue_R = E_R - ke_R;
            p_R = (gamma_R - 1.0) * ue_R - gamma_R * p0_R;
            h_TR = (ke_R + ue_R + p_R);
        }
        else
        {
            // With Respect to Specific Mass (energies are per unit mass)
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L * rho_L + small);
            ue_L = E_L - ke_L;
            p_L = (gamma_L - 1.0) * rho_L * ue_L - gamma_L * p0_L;
            h_TL = ke_L + ue_L + (p_L / rho_L);

            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R * rho_R + small);
            ue_R = E_R - ke_R;
            p_R = (gamma_R - 1.0) * rho_R * ue_R - gamma_R * p0_R;
            h_TR = ke_R + ue_R + (p_R / rho_R);
        }

        Set::Scalar u_L = Mn_L / (rho_L + small), u_R = Mn_R / (rho_R + small);
        Set::Scalar v_L = Mt_L / (rho_L + small), v_R = Mt_R / (rho_R + small);

        // Sound speeds (Tammann EOS)
        Set::Scalar a_L = std::sqrt(gamma_L * (p_L + p0_L) / (rho_L + small));
        Set::Scalar a_R = std::sqrt(gamma_R * (p_R + p0_R) / (rho_R + small));

        // Wave speed estimates
        Set::Scalar S_L = std::min(u_L - a_L, u_R - a_R);
        Set::Scalar S_R = std::max(u_L + a_L, u_R + a_R);

        // Middle wave speed (contact discontinuity)
        Set::Scalar S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + small);

        // Star state densities
        Set::Scalar rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star + small);
        Set::Scalar rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star + small);

        // Star state energies
        Set::Scalar E_star_L, E_star_R;

        // Fluxes
        Flux f_L, f_R;
        Flux f_star_L, f_star_R;

        if (Spec_Vol == 1)
        {
            // Energies per unit volume [J/m^3]
            E_star_L = rho_star_L * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small)));
            E_star_R = rho_star_R * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small)));

            // ---- Left ----
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (E_L + p_L);

            // ---- Right ----
            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (E_R + p_R);

            // - Left Star --
            f_star_L.mass = f_L.mass + S_L * (rho_star_L - rho_L);
            f_star_L.momentum_normal = f_L.momentum_normal + S_L * (rho_star_L * S_star - rho_L * u_L);
            f_star_L.momentum_tangent = f_L.momentum_tangent + S_L * (rho_star_L * v_L - rho_L * v_L);
            f_star_L.energy = f_L.energy + S_L * (E_star_L - E_L);

            // - Right Star --
            f_star_R.mass = f_R.mass + S_R * (rho_star_R - rho_R);
            f_star_R.momentum_normal = f_R.momentum_normal + S_R * (rho_star_R * S_star - rho_R * u_R);
            f_star_R.momentum_tangent = f_R.momentum_tangent + S_R * (rho_star_R * v_R - rho_R * v_R);
            f_star_R.energy = f_R.energy + S_R * (E_star_R - E_R);
        }
        else
        {
            // Energies per unit mass [J/kg]
            E_star_L = E_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small));
            E_star_R = E_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small));

            // ---- Left ----
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (E_L + p_L / (rho_L + small));

            // ---- Right ----
            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (E_R + p_R / (rho_R + small));

            // - Left Star --
            f_star_L.mass = f_L.mass + S_L * (rho_star_L - rho_L);
            f_star_L.momentum_normal = f_L.momentum_normal + S_L * (rho_star_L * S_star - rho_L * u_L);
            f_star_L.momentum_tangent = f_L.momentum_tangent + S_L * (rho_star_L * v_L - rho_L * v_L);
            f_star_L.energy = f_L.energy + S_L * ((E_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small))) - E_L);

            // - Right Star --
            f_star_R.mass = f_R.mass + S_R * (rho_star_R - rho_R);
            f_star_R.momentum_normal = f_R.momentum_normal + S_R * (rho_star_R * S_star - rho_R * u_R);
            f_star_R.momentum_tangent = f_R.momentum_tangent + S_R * (rho_star_R * v_R - rho_R * v_R);
            f_star_R.energy = f_R.energy + S_R * ((E_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small))) - E_R);
        }

        // Compute HLLC flux based on wave positions
        Flux fl;
        if (0.0 <= S_L)
        {
            fl = f_L;
        }
        else if (S_L <= 0.0 && 0.0 <= S_star)
        {
            fl = f_star_L;
        }
        else if (S_star <= 0.0 && 0.0 <= S_R)
        {
            fl = f_star_R;
        }
        else
        {
            fl = f_R;
        }

        // ERROR CHECKING
        if ((fl.mass != fl.mass) || (fl.momentum_normal != fl.momentum_normal) || (fl.momentum_tangent != fl.momentum_tangent) || (fl.energy != fl.energy) || (a_L != a_L) || (a_R != a_R))
        {
            bool verbose = false;
            if (verbose)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN HLLC_WENO5 SOLVER:");
                Util::ParallelMessage(INFO, "hi:   ", hi);
                Util::ParallelMessage(INFO, "lo:   ", lo);
                Util::ParallelMessage(INFO, "flux: ", fl);
                Util::ParallelMessage(INFO, "rho_L=", rho_L);
                Util::ParallelMessage(INFO, "rho_R=", rho_R);
                Util::ParallelMessage(INFO, "u_L=  ", u_L);
                Util::ParallelMessage(INFO, "u_R=  ", u_R);
                Util::ParallelMessage(INFO, "p_L=  ", p_L);
                Util::ParallelMessage(INFO, "p_R=  ", p_R);
                Util::ParallelMessage(INFO, "a_L=  ", a_L);
                Util::ParallelMessage(INFO, "a_R=  ", a_R);
                Util::ParallelMessage(INFO, "S_L=  ", S_L);
                Util::ParallelMessage(INFO, "S_R=  ", S_R);
                Util::ParallelMessage(INFO, "S_star=", S_star);
            }
            Util::Exception(INFO);
        }

        return fl;
    }

    // Solve for a domain with cell-averaged values
    std::vector<Flux> SolveArray(const std::vector<State> &cellAverages,
                                 Set::Scalar small,
                                 int Spec_Vol = 1)
    {
        const int n = cellAverages.size();
        std::vector<State> leftStates, rightStates;
        std::vector<Flux> interfaceFluxes(n + 1);

        // Perform WENO5 reconstruction
        if (use_characteristic_decomp)
        {
            weno_.reconstructCharacteristicStates(cellAverages, leftStates, rightStates, small, Spec_Vol);
        }
        else
        {
            weno_.reconstructStates(cellAverages, leftStates, rightStates);
        }

        // Solve Riemann problems at each interface
        for (int i = 0; i <= n; i++)
        {
            interfaceFluxes[i] = Solve(rightStates[i], leftStates[i], small, Spec_Vol);
        }

        return interfaceFluxes;
    }

    static int Test()
    {
        HLLC_WENO5 solver;
        int failed = 0;
        Set::Scalar small = 1E-10;
        int Spec_Vol = 1;

        // Test 1: Tangential Velocity Difference - No Normal Flux
        try
        {
            State left(1.0, 1.0, 0.0, 1.0);
            State center(1.0, 1.0, 1.0, 1.0);
            State right(1.0, 1.0, 2.0, 1.0);
            Flux fluxlo = solver.Solve(center, right, small, Spec_Vol);
            Flux fluxhi = solver.Solve(left, center, small, Spec_Vol);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10 || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:    ", left);
                Util::Warning(INFO, "center:  ", center);
                Util::Warning(INFO, "right:   ", right);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Exception(INFO, "Tangential velocity difference incorrectly affecting normal flux.");
            }
            Util::Test::SubMessage("Test 1: Tangential velocity should induce no normal flux", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 1: Tangential velocity should induce no normal flux", 1);
        }

        // Test 2: Pure Transverse Velocity Difference
        try
        {
            State left(1.0, 0.0, 0.0, 1.0);
            State center(1.0, 0.0, 1.0, 1.0);
            State right(1.0, 0.0, 2.0, 1.0);
            Flux fluxlo = solver.Solve(left, center, small, Spec_Vol);
            Flux fluxhi = solver.Solve(center, right, small, Spec_Vol);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10 || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "center: ", center);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Exception(INFO, "Pure transverse velocity difference affecting normal flux.");
            }
            Util::Test::SubMessage("Test 2: Pure transverse velocity difference", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 2: Pure transverse velocity difference", 1);
        }

        // Test 3: Symmetry Test (no flux across identical states)
        try
        {
            State left(1.0, 0.0, 0.0, 1.0);
            State center(1.0, 0.0, 0.0, 1.0);
            State right(1.0, 0.0, 0.0, 1.0);
            Flux fluxhi = solver.Solve(center, right, small, Spec_Vol);
            Flux fluxlo = solver.Solve(left, center, small, Spec_Vol);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10 || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 || fabs(fluxhi.momentum_tangent) > 1E-10 || fabs(fluxlo.momentum_tangent) > 1E-10 || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Exception(INFO, "Symmetric states should result in zero flux.");
            }
            Util::Test::SubMessage("Test 3: Constant states induces no flux difference", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 3: Constant states induces no flux difference", 1);
        }

        // Test 4: Uniform Flow Test
        try
        {
            State left(1.0, 1.0, 0.5, 1.0);
            State center(1.0, 1.0, 0.5, 1.0);
            State right(1.0, 1.0, 0.5, 1.0);
            Flux fluxhi = solver.Solve(center, right, small, Spec_Vol);
            Flux fluxlo = solver.Solve(left, center, small, Spec_Vol);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10 || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:   ", left);
                Util::Warning(INFO, "center: ", center);
                Util::Warning(INFO, "right:  ", right);
                Util::Warning(INFO, "Fluxlo: ", fluxlo);
                Util::Warning(INFO, "Fluxhi: ", fluxhi);
                Util::Exception(INFO, "Uniform flow should result in constant flux.");
            }
            Util::Test::SubMessage("Test 4: Uniform flow should maintain constant flux", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 4: Uniform flow should maintain constant flux", 1);
        }

        return failed;
    }

private:
    WENO5 weno_;
};

} // namespace FluidRiemann
} // namespace Local
} // namespace Solver

#endif // SOLVER_LOCAL_FLUIDRIEMANN_HLLC_WENO5_H
