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
// and Almost Incompressible Media," SIAM Journal on Scientific Computing, vol. 39, no. 3, pp. B486–B507,
// Jan. 2017, doi: 10.1137/16m1067834.
//          URL: https://arxiv.org/abs/1604.08544
//
//          If you wish to use the regular gas HLLC sovler and not the stiffened gas modeled by the Tammann EOS, set P0 = 0

#ifndef SOLVER_LOCAL_RIEMANN_HLLC_WENO5_H
#define SOLVER_LOCAL_RIEMANN_HLLC_WENO5_H

#include "IO/ParmParse.H"
#include "Solver/Local/Riemann/Riemann.H"
#include <cmath>
#include <vector>

/// A bunch of solvers
namespace Solver
{
/// Local solvers
namespace Local
{
namespace Riemann
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
    double
    reconstructLeft(const double *v, int i) const
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
        double w1 = w1t / sum;
        double w2 = w2t / sum;
        double w3 = w3t / sum;

        // Compute left interface value (right state at i-1/2)
        return w1 * s1 + w2 * s2 + w3 * s3;
    }

    double
    reconstructRight(const double *v, int i) const
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
        double w1 = w1t / sum;
        double w2 = w2t / sum;
        double w3 = w3t / sum;

        // Compute right interface value (left state at i+1/2)
        return w1 * s1 + w2 * s2 + w3 * s3;
    }

    // Reconstruct states at cell interfaces
    void
    reconstructStates(const std::vector<State> &cellAverages,
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

        // Add ghost cells (simple extrapolation for this example)
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
            // Adjust index for the extended array with ghost cells
            int idx = i + 2;
            if (i == 0)
            {
                // Left boundary
                leftStates[i].rho = rho[2];
                leftStates[i].M_normal = Mn[2];
                leftStates[i].M_tangent = Mt[2];
                leftStates[i].E = E[2];
                rightStates[i].rho = rho[2];
                rightStates[i].M_normal = Mn[2];
                rightStates[i].M_tangent = Mt[2];
                rightStates[i].E = E[2];
            }
            else if (i == n)
            {
                // Right boundary
                leftStates[i].rho = rho[n + 1];
                leftStates[i].M_normal = Mn[n + 1];
                leftStates[i].M_tangent = Mt[n + 1];
                leftStates[i].E = E[n + 1];
                rightStates[i].rho = rho[n + 1];
                rightStates[i].M_normal = Mn[n + 1];
                rightStates[i].M_tangent = Mt[n + 1];
                rightStates[i].E = E[n + 1];
            }
            else
            {
                // Interior interfaces
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
    void
    reconstructCharacteristicStates(const std::vector<State> &cellAverages,
                                    std::vector<State> &leftStates,
                                    std::vector<State> &rightStates,
                                    Set::Scalar gamma,
                                    Set::Scalar p_ref,
                                    Set::Scalar small,
                                    Set::Scalar p_0 = 0.0,
                                    Set::Scalar gamma_alt = 0.0,
                                    int Spec_Vol = 1) const
    {
        // Use gamma_alt if provided, otherwise use gamma
        Set::Scalar gamma_eff = (gamma_alt > 0.0) ? gamma_alt : gamma;

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
            int idx = i + 2; // Adjusted index for extended array
            if (i == 0 || i == n)
            {
                // Boundary interfaces - use simple extrapolation
                if (i == 0)
                {
                    leftStates[i] = extendedCells[2];
                    rightStates[i] = extendedCells[2];
                }
                else
                {
                    leftStates[i] = extendedCells[n + 1];
                    rightStates[i] = extendedCells[n + 1];
                }
            }
            else
            {
                // Interior interfaces - use characteristic decomposition
                // 1. Compute Roe average at interface
                Set::Scalar rho_L = extendedCells[idx - 1].rho;
                Set::Scalar rho_R = extendedCells[idx].rho;
                Set::Scalar sqrt_rho_L = std::sqrt(rho_L);
                Set::Scalar sqrt_rho_R = std::sqrt(rho_R);
                Set::Scalar u_L = extendedCells[idx - 1].M_normal / (rho_L + small);
                Set::Scalar u_R = extendedCells[idx].M_normal / (rho_R + small);
                Set::Scalar v_L = extendedCells[idx - 1].M_tangent / (rho_L + small);
                Set::Scalar v_R = extendedCells[idx].M_tangent / (rho_R + small);
                Set::Scalar E_L = extendedCells[idx - 1].E;
                Set::Scalar E_R = extendedCells[idx].E;

                // Calculate kinetic energy and pressure based on Spec_Vol
                Set::Scalar ke_L, ke_R, p_L, p_R;

                if (Spec_Vol == 1)
                {
                    // Volume-specific energy
                    ke_L = 0.5 * (u_L * u_L + v_L * v_L) * rho_L;
                    ke_R = 0.5 * (u_R * u_R + v_R * v_R) * rho_R;
                    p_L = (gamma_eff - 1.0) * (E_L - ke_L) - gamma_eff * p_0 + p_ref;
                    p_R = (gamma_eff - 1.0) * (E_R - ke_R) - gamma_eff * p_0 + p_ref;
                }
                else
                {
                    // Mass-specific energy
                    ke_L = 0.5 * (u_L * u_L + v_L * v_L);
                    ke_R = 0.5 * (u_R * u_R + v_R * v_R);
                    p_L = (gamma_eff - 1.0) * rho_L * (E_L - ke_L) - gamma_eff * p_0 + p_ref;
                    p_R = (gamma_eff - 1.0) * rho_R * (E_R - ke_R) - gamma_eff * p_0 + p_ref;
                }

                // Calculate enthalpy based on Spec_Vol
                Set::Scalar H_L, H_R;
                if (Spec_Vol == 1)
                {
                    // Volume-specific energy
                    H_L = (E_L + p_L) / rho_L;
                    H_R = (E_R + p_R) / rho_R;
                }
                else
                {
                    // Mass-specific energy
                    H_L = E_L + p_L / rho_L;
                    H_R = E_R + p_R / rho_R;
                }

                // Roe averages
                Set::Scalar rho_roe = sqrt_rho_L * sqrt_rho_R;
                Set::Scalar u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R + small);
                Set::Scalar v_roe = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) / (sqrt_rho_L + sqrt_rho_R + small);
                Set::Scalar H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R + small);

                // Sound speed calculation for Tammann EOS
                Set::Scalar p_roe = (sqrt_rho_L * p_L + sqrt_rho_R * p_R) / (sqrt_rho_L + sqrt_rho_R + small);
                Set::Scalar a_roe = std::sqrt(gamma_eff * (p_roe + p_0) / rho_roe);

                // 2. Compute eigenvectors
                // For 1D Euler with tangential velocity, we have 4 characteristic fields
                // Right eigenvectors matrix (R)
                double R[4][4] = {
                    { 1.0, 1.0, 1.0, 0.0 },
                    { u_roe - a_roe, u_roe, u_roe + a_roe, 0.0 },
                    { v_roe, v_roe, v_roe, 1.0 },
                    { H_roe - u_roe * a_roe, 0.5 * (u_roe * u_roe + v_roe * v_roe), H_roe + u_roe * a_roe, v_roe }
                };

                // Left eigenvectors matrix (L = R^-1)
                double b1 = (gamma_eff - 1.0) / (a_roe * a_roe);
                double b2 = 0.5 * (u_roe * u_roe + v_roe * v_roe);
                double L[4][4] = {
                    { 0.5 * (b1 * b2 + u_roe / a_roe), -0.5 * (b1 * u_roe + 1.0 / a_roe), -0.5 * b1 * v_roe, 0.5 * b1 },
                    { 1.0 - b1 * b2, b1 * u_roe, b1 * v_roe, -b1 },
                    { 0.5 * (b1 * b2 - u_roe / a_roe), -0.5 * (b1 * u_roe - 1.0 / a_roe), -0.5 * b1 * v_roe, 0.5 * b1 },
                    { -v_roe, 0.0, 1.0, 0.0 }
                };

                // 3. Project to characteristic variables
                std::vector<std::vector<double> > charVars(4, std::vector<double>(5));
                for (int j = idx - 3; j <= idx + 1; j++)
                {
                    // Convert state to conserved variables
                    double U[4] = {
                        extendedCells[j].rho,
                        extendedCells[j].M_normal,
                        extendedCells[j].M_tangent,
                        extendedCells[j].E
                    };

                    // Project to characteristic variables
                    for (int k = 0; k < 4; k++)
                    {
                        charVars[k][j - (idx - 3)] = 0.0;
                        for (int m = 0; m < 4; m++)
                        {
                            charVars[k][j - (idx - 3)] += L[k][m] * U[m];
                        }
                    }
                }

                // 4. Apply WENO5 to each characteristic field
                double leftChar[4], rightChar[4];
                for (int k = 0; k < 4; k++)
                {
                    rightChar[k] = reconstructRight(charVars[k].data(), 3);
                    leftChar[k] = reconstructLeft(charVars[k].data(), 2);
                }

                // 5. Project back to physical space
                double leftU[4] = { 0.0 }, rightU[4] = { 0.0 };
                for (int k = 0; k < 4; k++)
                {
                    for (int m = 0; m < 4; m++)
                    {
                        leftU[k] += R[k][m] * leftChar[m];
                        rightU[k] += R[k][m] * rightChar[m];
                    }
                }

                // 6. Convert back to State
                leftStates[i].rho = leftU[0];
                leftStates[i].M_normal = leftU[1];
                leftStates[i].M_tangent = leftU[2];
                leftStates[i].E = leftU[3];
                rightStates[i].rho = rightU[0];
                rightStates[i].M_normal = rightU[1];
                rightStates[i].M_tangent = rightU[2];
                rightStates[i].E = rightU[3];

                // 7. Ensure physical values (non-negative density and pressure)
                leftStates[i].rho = std::max(small, leftStates[i].rho);
                rightStates[i].rho = std::max(small, rightStates[i].rho);

                // Recalculate energy if needed to ensure positive pressure
                if (Spec_Vol == 1)
                {
                    // Volume-specific energy
                    Set::Scalar ke_left = 0.5 * (leftStates[i].M_normal * leftStates[i].M_normal + leftStates[i].M_tangent * leftStates[i].M_tangent) / (leftStates[i].rho + small);
                    Set::Scalar p_left = (gamma_eff - 1.0) * (leftStates[i].E - ke_left) - gamma_eff * p_0 + p_ref;
                    if (p_left < small)
                    {
                        leftStates[i].E = (small + gamma_eff * p_0 - p_ref) / (gamma_eff - 1.0) + ke_left;
                    }

                    Set::Scalar ke_right = 0.5 * (rightStates[i].M_normal * rightStates[i].M_normal + rightStates[i].M_tangent * rightStates[i].M_tangent) / (rightStates[i].rho + small);
                    Set::Scalar p_right = (gamma_eff - 1.0) * (rightStates[i].E - ke_right) - gamma_eff * p_0 + p_ref;
                    if (p_right < small)
                    {
                        rightStates[i].E = (small + gamma_eff * p_0 - p_ref) / (gamma_eff - 1.0) + ke_right;
                    }
                }
                else
                {
                    // Mass-specific energy
                    Set::Scalar ke_left = 0.5 * (leftStates[i].M_normal * leftStates[i].M_normal + leftStates[i].M_tangent * leftStates[i].M_tangent) / (leftStates[i].rho * leftStates[i].rho + small);
                    Set::Scalar p_left = (gamma_eff - 1.0) * leftStates[i].rho * (leftStates[i].E - ke_left) - gamma_eff * p_0 + p_ref;
                    if (p_left < small)
                    {
                        leftStates[i].E = (small + gamma_eff * p_0 - p_ref) / ((gamma_eff - 1.0) * leftStates[i].rho) + ke_left;
                    }

                    Set::Scalar ke_right = 0.5 * (rightStates[i].M_normal * rightStates[i].M_normal + rightStates[i].M_tangent * rightStates[i].M_tangent) / (rightStates[i].rho * rightStates[i].rho + small);
                    Set::Scalar p_right = (gamma_eff - 1.0) * rightStates[i].rho * (rightStates[i].E - ke_right) - gamma_eff * p_0 + p_ref;
                    if (p_right < small)
                    {
                        rightStates[i].E = (small + gamma_eff * p_0 - p_ref) / ((gamma_eff - 1.0) * rightStates[i].rho) + ke_right;
                    }
                }
            }
        }
    }

private:
    double eps_; // Small parameter to avoid division by zero
};

/// HLLC Riemann Solver with WENO5 reconstruction
class HLLC_WENO5
{
public:
    static constexpr const char *name = "hllc_weno5";

    HLLC_WENO5(IO::ParmParse &pp, std::string name)
    {
        pp_queryclass(name, *this);
    }

    HLLC_WENO5(IO::ParmParse &pp)
    {
        pp_queryclass(*this);
    }

    HLLC_WENO5()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    int verbose = 0;
    bool use_characteristic_decomp = true;

    static void
    Parse(HLLC_WENO5 &value, IO::ParmParse &pp)
    {
        // enable to dump diagnostic data if the hllc solver fails
        pp.query_default("verbose", value.verbose, 1);
        // use characteristic decomposition for WENO reconstruction
        pp.query_default("use_characteristic_decomp", value.use_characteristic_decomp, true);
    }

    // Solve for a single interface with given left and right states
    Flux
    SolveInterface(State lo, State hi, Set::Scalar gamma, Set::Scalar p_ref, Set::Scalar small, Set::Scalar p_0 = 0.0, Set::Scalar gamma_alt = 0.0, int Spec_Vol = 1)
    {
        // Use gamma_alt if provided, otherwise use gamma
        Set::Scalar gamma_eff = (gamma_alt > 0.0) ? gamma_alt : gamma;

        Set::Scalar rho_L = lo.rho, rho_R = hi.rho;
        Set::Scalar Mn_L = lo.M_normal, Mn_R = hi.M_normal;
        Set::Scalar Mt_L = lo.M_tangent, Mt_R = hi.M_tangent;
        Set::Scalar E_L = lo.E, E_R = hi.E;

        // Ensure no negative densities
        rho_L = std::max(small, rho_L);
        rho_R = std::max(small, rho_R);

        // STEP 1: Compute fluid primitives
        Set::Scalar ke_L, ke_R, ue_L, ue_R, p_L, p_R, h_TL, h_TR;

        if (Spec_Vol == 1)
        {
            // With Respect to Specific Volume (energies are per unit volume)
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L + small); // KE per unit volume
            ue_L = E_L - ke_L;                                          // IE per unit volume
            p_L = (gamma_eff - 1.0) * ue_L - gamma_eff * p_0 + p_ref;   // pressure Tammann EOS modification
            h_TL = (ke_L + ue_L + p_L);                                 // stagnation enthalpy (per unit volume)

            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R + small);
            ue_R = E_R - ke_R;
            p_R = (gamma_eff - 1.0) * ue_R - gamma_eff * p_0 + p_ref;
            h_TR = (ke_R + ue_R + p_R);
        }
        else
        {
            // With Respect to Specific Mass (energies are per unit mass)
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L * rho_L + small); // KE per unit mass
            ue_L = E_L - ke_L;                                                  // IE per unit mass
            p_L = (gamma_eff - 1.0) * rho_L * ue_L - gamma_eff * p_0 + p_ref;   // pressure Tammann EOS modification
            h_TL = ke_L + ue_L + (p_L) / (rho_L + small);                       // specific stagnation enthalpy (per unit mass)

            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R * rho_R + small);
            ue_R = E_R - ke_R;
            p_R = (gamma_eff - 1.0) * rho_R * ue_R - gamma_eff * p_0 + p_ref;
            h_TR = ke_R + ue_R + (p_R) / (rho_R + small);
        }

        Set::Scalar u_L = Mn_L / (rho_L + small), u_R = Mn_R / (rho_R + small);
        Set::Scalar v_L = Mt_L / (rho_L + small), v_R = Mt_R / (rho_R + small);

        // STEP 2: Compute sound speeds for Tammann EOS
        Set::Scalar a_L = std::sqrt(gamma_eff * (p_L + p_0) / (rho_L + small));
        Set::Scalar a_R = std::sqrt(gamma_eff * (p_R + p_0) / (rho_R + small));

        // Ensure sound speeds are positive
        a_L = std::max(small, a_L);
        a_R = std::max(small, a_R);

        // STEP 3: Compute wave speed estimates
        // Use Davis estimates for wave speeds
        Set::Scalar S_L = std::min(u_L - a_L, u_R - a_R);
        Set::Scalar S_R = std::max(u_L + a_L, u_R + a_R);

        // STEP 4: Compute middle wave speed (contact discontinuity)
        Set::Scalar S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + small);

        // STEP 5: Compute physical fluxes for left and right states
        Flux f_L, f_R;

        if (Spec_Vol == 1)
        {
            // Volume-specific energy fluxes
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (E_L + p_L);

            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (E_R + p_R);
        }
        else
        {
            // Mass-specific energy fluxes
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (rho_L * E_L + p_L);

            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (rho_R * E_R + p_R);
        }

        // STEP 6: Compute star state values
        Set::Scalar rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star + small);
        Set::Scalar rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star + small);

        // Ensure star densities are positive
        rho_star_L = std::max(small, rho_star_L);
        rho_star_R = std::max(small, rho_star_R);

        // Compute star pressure (consistent with HLLC formulation)
        Set::Scalar p_star = rho_L * (S_L - u_L) * (S_star - u_L) + p_L;

        // Ensure star pressure is positive
        p_star = std::max(small, p_star);

        // STEP 7: Compute star state fluxes
        Flux f_star_L, f_star_R;

        if (Spec_Vol == 1)
        {
            // Volume-specific energy star fluxes
            f_star_L.mass = f_L.mass + S_L * (rho_star_L - rho_L);
            f_star_L.momentum_normal = f_L.momentum_normal + S_L * (rho_star_L * S_star - rho_L * u_L);
            f_star_L.momentum_tangent = f_L.momentum_tangent + S_L * (rho_star_L * v_L - rho_L * v_L);
            f_star_L.energy = f_L.energy + S_L * (rho_star_L * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small))) - E_L);

            f_star_R.mass = f_R.mass + S_R * (rho_star_R - rho_R);
            f_star_R.momentum_normal = f_R.momentum_normal + S_R * (rho_star_R * S_star - rho_R * u_R);
            f_star_R.momentum_tangent = f_R.momentum_tangent + S_R * (rho_star_R * v_R - rho_R * v_R);
            f_star_R.energy = f_R.energy + S_R * (rho_star_R * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small))) - E_R);
        }
        else
        {
            // Mass-specific energy star fluxes
            f_star_L.mass = f_L.mass + S_L * (rho_star_L - rho_L);
            f_star_L.momentum_normal = f_L.momentum_normal + S_L * (rho_star_L * S_star - rho_L * u_L);
            f_star_L.momentum_tangent = f_L.momentum_tangent + S_L * (rho_star_L * v_L - rho_L * v_L);
            f_star_L.energy = f_L.energy + S_L * (rho_star_L * E_L + rho_star_L * (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small)) - rho_L * E_L);

            f_star_R.mass = f_R.mass + S_R * (rho_star_R - rho_R);
            f_star_R.momentum_normal = f_R.momentum_normal + S_R * (rho_star_R * S_star - rho_R * u_R);
            f_star_R.momentum_tangent = f_R.momentum_tangent + S_R * (rho_star_R * v_R - rho_R * v_R);
            f_star_R.energy = f_R.energy + S_R * (rho_star_R * E_R + rho_star_R * (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small)) - rho_R * E_R);
        }

        // STEP 8: Compute HLLC flux based on wave positions
        Flux fl;

        if (0 <= S_L)
        {
            fl = f_L;
        }
        else if (S_L <= 0 && 0 <= S_star)
        {
            fl = f_star_L;
        }
        else if (S_star <= 0 && 0 <= S_R)
        {
            fl = f_star_R;
        }
        else
        {
            fl = f_R;
        }

        // ERROR CHECKING
        if ((fl.mass != fl.mass) || (fl.momentum_normal != fl.momentum_normal) || (fl.momentum_tangent != fl.momentum_tangent) || (fl.energy != fl.energy))
        {

            if (verbose)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN HLLC_WENO5 SOLVER:");
                Util::ParallelMessage(INFO, "hi ", hi);
                Util::ParallelMessage(INFO, "lo ", lo);
                Util::ParallelMessage(INFO, "rho_R ", rho_R);
                Util::ParallelMessage(INFO, "rho_L ", rho_L);
                Util::ParallelMessage(INFO, "u_R ", u_R);
                Util::ParallelMessage(INFO, "u_L ", u_L);
                Util::ParallelMessage(INFO, "p_R ", p_R);
                Util::ParallelMessage(INFO, "p_L ", p_L);
                Util::ParallelMessage(INFO, "a_R ", a_R);
                Util::ParallelMessage(INFO, "a_L ", a_L);
                Util::ParallelMessage(INFO, "S_L ", S_L);
                Util::ParallelMessage(INFO, "S_R ", S_R);
                Util::ParallelMessage(INFO, "S_star ", S_star);
                Util::ParallelMessage(INFO, "Spec_Vol ", Spec_Vol);
            }

            // Fallback to a more robust flux calculation
            if (0 <= S_L)
            {
                fl = f_L;
            }
            else if (S_R <= 0)
            {
                fl = f_R;
            }
            else
            {
                // HLL flux as fallback
                fl.mass = (S_R * f_L.mass - S_L * f_R.mass + S_L * S_R * (rho_R - rho_L)) / (S_R - S_L + small);
                fl.momentum_normal = (S_R * f_L.momentum_normal - S_L * f_R.momentum_normal + S_L * S_R * (rho_R * u_R - rho_L * u_L)) / (S_R - S_L + small);
                fl.momentum_tangent = (S_R * f_L.momentum_tangent - S_L * f_R.momentum_tangent + S_L * S_R * (rho_R * v_R - rho_L * v_L)) / (S_R - S_L + small);
                fl.energy = (S_R * f_L.energy - S_L * f_R.energy + S_L * S_R * (E_R - E_L)) / (S_R - S_L + small);
            }

            // Check if fallback flux is valid
            if ((fl.mass != fl.mass) || (fl.momentum_normal != fl.momentum_normal) || (fl.momentum_tangent != fl.momentum_tangent) || (fl.energy != fl.energy))
            {

                // Last resort: use upwind flux
                if (u_L + u_R >= 0)
                {
                    fl = f_L;
                }
                else
                {
                    fl = f_R;
                }

                Util::ParallelMessage(INFO, "WARNING: Using upwind flux as last resort");
            }
        }

        return fl;
    }

    // Solve for a domain with cell-averaged values
    std::vector<Flux>
    Solve(const std::vector<State> &cellAverages,
          Set::Scalar gamma,
          Set::Scalar p_ref,
          Set::Scalar small,
          Set::Scalar p_0 = 0.0,
          Set::Scalar gamma_alt = 0.0,
          int Spec_Vol = 1)
    {
        const int n = cellAverages.size();
        std::vector<State> leftStates, rightStates;
        std::vector<Flux> interfaceFluxes(n + 1);

        // Perform WENO5 reconstruction
        if (use_characteristic_decomp)
        {
            weno_.reconstructCharacteristicStates(cellAverages, leftStates, rightStates, gamma, p_ref, small, p_0, gamma_alt, Spec_Vol);
        }
        else
        {
            weno_.reconstructStates(cellAverages, leftStates, rightStates);
        }

        // Solve Riemann problems at each interface
        for (int i = 0; i <= n; i++)
        {
            interfaceFluxes[i] = SolveInterface(rightStates[i], leftStates[i], gamma, p_ref, small, p_0, gamma_alt, Spec_Vol);
        }

        return interfaceFluxes;
    }

    // Original interface for compatibility with your Hydro2 solver
    Flux
    Solve(State lo, State hi, Set::Scalar gamma, Set::Scalar p_ref, Set::Scalar small, Set::Scalar p_0 = 0.0, Set::Scalar gamma_alt = 0.0, int Spec_Vol = 1)
    {
        return SolveInterface(lo, hi, gamma, p_ref, small, p_0, gamma_alt, Spec_Vol);
    }

    static int
    Test()
    {
        HLLC_WENO5 solver;
        int failed = 0;
        Set::Scalar gamma = 1.4;
        Set::Scalar pref = 10.0;
        Set::Scalar small = 1E-10;
        Set::Scalar p0 = 0.0;
        int Spec_Vol = 1;

        // Test 1: Tangential Velocity Difference - No Normal Flux
        try
        {
            State left(1.0, 1.0, 0.0, 1.0);
            State center(1.0, 1.0, 1.0, 1.0);
            State right(1.0, 1.0, 2.0, 1.0);
            Flux fluxlo = solver.Solve(center, right, gamma, pref, small, p0, 0.0, Spec_Vol);
            Flux fluxhi = solver.Solve(left, center, gamma, pref, small, p0, 0.0, Spec_Vol);

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
            Flux fluxlo = solver.Solve(left, center, gamma, pref, small, p0, 0.0, Spec_Vol);
            Flux fluxhi = solver.Solve(center, right, gamma, pref, small, p0, 0.0, Spec_Vol);

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
            Flux fluxhi = solver.Solve(center, right, gamma, pref, small, p0, 0.0, Spec_Vol);
            Flux fluxlo = solver.Solve(left, center, gamma, pref, small, p0, 0.0, Spec_Vol);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10 ||                       // no change in mass flux
                fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 || // no change in momentum flux
                fabs(fluxhi.momentum_tangent) > 1E-10 ||                         // zero tangent flux
                fabs(fluxlo.momentum_tangent) > 1E-10 ||                         // zero tangent flux
                fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            { // no change in energy flux

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

        // Test 4: Uniform Flow Test (no flux across uniform flow)
        try
        {
            State left(1.0, 1.0, 0.5, 1.0);
            State center(1.0, 1.0, 0.5, 1.0);
            State right(1.0, 1.0, 0.5, 1.0);
            Flux fluxhi = solver.Solve(center, right, gamma, pref, small, p0, 0.0, Spec_Vol);
            Flux fluxlo = solver.Solve(left, center, gamma, pref, small, p0, 0.0, Spec_Vol);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10 || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {

                Util::Warning(INFO, "left:   ", left);
                Util::Warning(INFO, "center: ", center);
                Util::Warning(INFO, "right:  ", right);
                Util::Warning(INFO, "Fluxlo: ", fluxlo);
                Util::Warning(INFO, "Fluxhi: ", fluxhi);
                Util::Exception(INFO, "Uniform flow should result in no flux.");
            }

            Util::Test::SubMessage("Test 4: Uniform flow should maintain constant flux", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 4: Uniform flow should maintain constant flux", 1);
        }

        // Test 5: Tammann EOS Test (water-like fluid)
        try
        {
            Set::Scalar p0_water = 3.0e8;
            Set::Scalar gamma_water = 7.15;

            // Create states for water at different pressures
            State water_low(1000.0, 0.0, 0.0, 4.0e5);  // Water at low pressure
            State water_high(1000.0, 0.0, 0.0, 1.0e6); // Water at high pressure

            // Solve with Tammann EOS
            Flux flux = solver.Solve(water_low, water_high, gamma_water, 0.0, small, p0_water, 0.0, Spec_Vol);

            // Check that flux is not NaN
            if (flux.mass != flux.mass || flux.momentum_normal != flux.momentum_normal || flux.energy != flux.energy)
            {

                Util::Warning(INFO, "water_low:  ", water_low);
                Util::Warning(INFO, "water_high: ", water_high);
                Util::Warning(INFO, "Flux:       ", flux);
                Util::Exception(INFO, "Tammann EOS test produced NaN values.");
            }

            Util::Test::SubMessage("Test 5: Tammann EOS for water-like fluid", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 5: Tammann EOS for water-like fluid", 1);
        }

        return failed;
    }

private:
    WENO5 weno_;
};

} // namespace Riemann
} // namespace Local
} // namespace Solver

#endif // SOLVER_LOCAL_RIEMANN_HLLC_WENO5_H
