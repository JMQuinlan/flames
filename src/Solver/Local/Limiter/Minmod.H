#ifndef SOLVER_LOCAL_LIMITER_MINMOD_H
#define SOLVER_LOCAL_LIMITER_MINMOD_H

#include "IO/ParmParse.H"
#include "Solver/Local/Riemann/Riemann.H"
//#include <algorithm>
//#include <cmath>
//#include <vector>

/// Limiters for slope reconstruction
namespace Solver
{
/// Local solvers
namespace Local
{
/// Limiters
namespace Limiter
{

/// Minmod Limiter Class
class Minmod
{
public:
    using State = Solver::Local::Riemann::State;

    static constexpr const char *name = "minmod";

    Minmod(IO::ParmParse &pp, std::string name)
    {
        pp_queryclass(name, *this);
    }

    Minmod(IO::ParmParse &pp)
    {
        pp_queryclass(*this);
    }

    Minmod()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    int verbose = 0;
    Set::Scalar theta = 1.0; // Parameter to control dissipation (1.0 = standard minmod)

    static void
    Parse(Minmod &value, IO::ParmParse &pp)
    {
        // Enable to dump diagnostic data if the limiter fails
        pp.query_default("verbose", value.verbose, 1);

        // Theta parameter (1.0 = standard minmod, 2.0 = more compressive)
        pp.query_default("theta", value.theta, 1.0);

        // Ensure theta is in valid range
        value.theta = std::max(1.0, std::min(2.0, value.theta));
    }

    /// Basic minmod function for two arguments
    Set::Scalar
    minmod2(Set::Scalar a, Set::Scalar b) const
    {
        if (a * b <= 0.0)
            return 0.0; // Different signs or one is zero
        return (std::abs(a) < std::abs(b)) ? a : b;
    }

    /// Minmod function for three arguments
    Set::Scalar
    minmod3(Set::Scalar a, Set::Scalar b, Set::Scalar c) const
    {
        if ((a * b <= 0.0) || (a * c <= 0.0))
            return 0.0; // Different signs
        return (std::abs(a) < std::abs(b))
                   ? ((std::abs(a) < std::abs(c)) ? a : c)
                   : ((std::abs(b) < std::abs(c)) ? b : c);
    }

    /// Apply minmod limiter to a slope
    Set::Scalar
    apply(Set::Scalar slope_left, Set::Scalar slope_center, Set::Scalar slope_right) const
    {
        return minmod3(theta * slope_left, slope_center, theta * slope_right);
    }

    /// Apply minmod limiter to compute limited slope at cell i
    Set::Scalar
    limitSlope(const std::vector<Set::Scalar> &values, int i) const
    {
        if (i <= 0 || i >= static_cast<int>(values.size()) - 1)
        {
            // Boundary cell - return zero slope for safety
            return 0.0;
        }

        // Compute forward and backward differences
        Set::Scalar forward_diff = values[i + 1] - values[i];
        Set::Scalar backward_diff = values[i] - values[i - 1];
        Set::Scalar centered_diff = 0.5 * (values[i + 1] - values[i - 1]);

        // Apply minmod limiter
        return minmod3(theta * backward_diff, centered_diff, theta * forward_diff);
    }

    /// Reconstruct left and right states at cell interfaces using minmod limiter
    void
    reconstructStates(const std::vector<State> &cellAverages,
                      std::vector<State> &leftStates,
                      std::vector<State> &rightStates) const
    {
        const int n = cellAverages.size();
        leftStates.resize(n + 1);
        rightStates.resize(n + 1);

        // Extract components for reconstruction
        std::vector<Set::Scalar> rho(n), Mn(n), Mt(n), E(n);
        for (int i = 0; i < n; i++)
        {
            rho[i] = cellAverages[i].rho;
            Mn[i] = cellAverages[i].M_normal;
            Mt[i] = cellAverages[i].M_tangent;
            E[i] = cellAverages[i].E;
        }

        // Add ghost cells (simple extrapolation for this example)
        rho.insert(rho.begin(), rho[0]);
        rho.push_back(rho[n - 1]);
        Mn.insert(Mn.begin(), Mn[0]);
        Mn.push_back(Mn[n - 1]);
        Mt.insert(Mt.begin(), Mt[0]);
        Mt.push_back(Mt[n - 1]);
        E.insert(E.begin(), E[0]);
        E.push_back(E[n - 1]);

        // Compute limited slopes for each variable
        std::vector<Set::Scalar> rho_slopes(n), Mn_slopes(n), Mt_slopes(n), E_slopes(n);
        for (int i = 0; i < n; i++)
        {
            rho_slopes[i] = limitSlope(rho, i + 1); // +1 for ghost cell offset
            Mn_slopes[i] = limitSlope(Mn, i + 1);
            Mt_slopes[i] = limitSlope(Mt, i + 1);
            E_slopes[i] = limitSlope(E, i + 1);
        }

        // Reconstruct states at interfaces
        for (int i = 0; i <= n; i++)
        {
            if (i == 0)
            {
                // Left boundary
                leftStates[i] = cellAverages[0];
                rightStates[i] = cellAverages[0];
            }
            else if (i == n)
            {
                // Right boundary
                leftStates[i] = cellAverages[n - 1];
                rightStates[i] = cellAverages[n - 1];
            }
            else
            {
                // Interior interfaces
                // Right state at i-1/2 (left side of cell i)
                rightStates[i].rho          = rho[i] + 0.5 * rho_slopes[i - 1];
                rightStates[i].M_normal     = Mn[i] + 0.5 * Mn_slopes[i - 1];
                rightStates[i].M_tangent    = Mt[i] + 0.5 * Mt_slopes[i - 1];
                rightStates[i].E            = E[i] + 0.5 * E_slopes[i - 1];

                // Left state at i+1/2 (right side of cell i)
                leftStates[i].rho           = rho[i + 1] - 0.5 * rho_slopes[i];
                leftStates[i].M_normal      = Mn[i + 1] - 0.5 * Mn_slopes[i];
                leftStates[i].M_tangent     = Mt[i + 1] - 0.5 * Mt_slopes[i];
                leftStates[i].E             = E[i + 1] - 0.5 * E_slopes[i];
            }
        }
    }

    /// Apply limiter to characteristic variables (for systems of equations)
    void
    reconstructCharacteristicStates(const std::vector<State> &cellAverages,
                                    std::vector<State> &leftStates,
                                    std::vector<State> &rightStates,
                                    Set::Scalar gamma,
                                    Set::Scalar p_ref,
                                    Set::Scalar small,
                                    Set::Scalar p_0 = 0.0) const
    {
        const int n = cellAverages.size();
        leftStates.resize(n + 1);
        rightStates.resize(n + 1);

        // Add ghost cells
        std::vector<State> extendedCells = cellAverages;
        extendedCells.insert(extendedCells.begin(), cellAverages[0]);
        extendedCells.push_back(cellAverages[n - 1]);

        // For each interface
        for (int i = 0; i <= n; i++)
        {
            if (i == 0 || i == n)
            {
                // Boundary interfaces - use cell averages
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
                // Interior interfaces - use characteristic decomposition
                // Get the three cells needed for reconstruction
                State &left = extendedCells[i - 1];
                State &center = extendedCells[i];
                State &right = extendedCells[i + 1];

                // Compute primitive variables for center cell
                Set::Scalar rho_c = center.rho;
                Set::Scalar u_c = center.M_normal / (rho_c + small);
                Set::Scalar v_c = center.M_tangent / (rho_c + small);
                Set::Scalar ke_c = 0.5 * (u_c * u_c + v_c * v_c) * rho_c;
                Set::Scalar p_c = (gamma - 1.0) * (center.E - ke_c) - gamma * (p_0 + p_ref);
                Set::Scalar a_c = std::sqrt(gamma * (p_c + p_ref)/ (rho_c + small));

                // Compute eigenvectors at center cell
                // Right eigenvectors matrix (R)
                double R[4][4] = {
                    { 1.0, 1.0, 1.0, 0.0 },
                    { u_c - a_c, u_c, u_c + a_c, 0.0 },
                    { v_c, v_c, v_c, 1.0 },
                    { center.E / (rho_c + small) - u_c * a_c, 0.5 * (u_c * u_c + v_c * v_c), center.E / (rho_c + small) + u_c * a_c, v_c }
                };

                // Left eigenvectors matrix (L = R^-1)
                double b1 = (gamma - 1.0) / (a_c * a_c + small);
                double b2 = 0.5 * (u_c * u_c + v_c * v_c);
                double L[4][4] = {
                    { 0.5 * (b1 * b2 + u_c / (a_c + small)), -0.5 * (b1 * u_c + 1.0 / (a_c + small)), -0.5 * b1 * v_c, 0.5 * b1 },
                    { 1.0 - b1 * b2, b1 * u_c, b1 * v_c, -b1 },
                    { 0.5 * (b1 * b2 - u_c / (a_c + small)), -0.5 * (b1 * u_c - 1.0 / (a_c + small)), -0.5 * b1 * v_c, 0.5 * b1 },
                    { -v_c, 0.0, 1.0, 0.0 }
                };

                // Convert states to conserved variables
                double U_left[4] = {
                    left.rho, left.M_normal, left.M_tangent, left.E
                };
                double U_center[4] = {
                    center.rho, center.M_normal, center.M_tangent, center.E
                };
                double U_right[4] = {
                    right.rho, right.M_normal, right.M_tangent, right.E
                };

                // Project to characteristic variables
                double W_left[4] = { 0.0 }, W_center[4] = { 0.0 }, W_right[4] = { 0.0 };
                for (int k = 0; k < 4; k++)
                {
                    for (int m = 0; m < 4; m++)
                    {
                        W_left[k] += L[k][m] * U_left[m];
                        W_center[k] += L[k][m] * U_center[m];
                        W_right[k] += L[k][m] * U_right[m];
                    }
                }

                // Compute limited slopes in characteristic space
                double W_slopes[4] = { 0.0 };
                for (int k = 0; k < 4; k++)
                {
                    double forward_diff = W_right[k] - W_center[k];
                    double backward_diff = W_center[k] - W_left[k];
                    double centered_diff = 0.5 * (W_right[k] - W_left[k]);
                    W_slopes[k] = minmod3(theta * backward_diff, centered_diff, theta * forward_diff);
                }

                // Reconstruct characteristic variables at interface
                double W_left_interface[4] = { 0.0 }, W_right_interface[4] = { 0.0 };
                for (int k = 0; k < 4; k++)
                {
                    W_left_interface[k] = W_center[k] - 0.5 * W_slopes[k];
                    W_right_interface[k] = W_center[k] + 0.5 * W_slopes[k];
                }

                // Project back to conserved variables
                double U_left_interface[4] = { 0.0 }, U_right_interface[4] = { 0.0 };
                for (int k = 0; k < 4; k++)
                {
                    for (int m = 0; m < 4; m++)
                    {
                        U_left_interface[k] += R[k][m] * W_left_interface[m];
                        U_right_interface[k] += R[k][m] * W_right_interface[m];
                    }
                }

                // Set interface states
                rightStates[i].rho = U_left_interface[0];
                rightStates[i].M_normal = U_left_interface[1];
                rightStates[i].M_tangent = U_left_interface[2];
                rightStates[i].E = U_left_interface[3];

                leftStates[i].rho = U_right_interface[0];
                leftStates[i].M_normal = U_right_interface[1];
                leftStates[i].M_tangent = U_right_interface[2];
                leftStates[i].E = U_right_interface[3];
            }
        }
    }

    static int
    Test()
    {
        Minmod limiter;
        int failed = 0;

        // Test 1: Basic minmod function
        try
        {
            // Same sign, positive
            if (std::abs(limiter.minmod2(1.0, 2.0) - 1.0) > 1E-10)
            {
                Util::Exception(INFO, "minmod(1.0, 2.0) should be 1.0");
            }

            // Same sign, negative
            if (std::abs(limiter.minmod2(-1.0, -2.0) - (-1.0)) > 1E-10)
            {
                Util::Exception(INFO, "minmod(-1.0, -2.0) should be -1.0");
            }

            // Different signs
            if (std::abs(limiter.minmod2(1.0, -2.0)) > 1E-10)
            {
                Util::Exception(INFO, "minmod(1.0, -2.0) should be 0.0");
            }

            // One zero
            if (std::abs(limiter.minmod2(0.0, 2.0)) > 1E-10)
            {
                Util::Exception(INFO, "minmod(0.0, 2.0) should be 0.0");
            }

            Util::Test::SubMessage("Test 1: Basic minmod function", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 1: Basic minmod function", 1);
        }

        // Test 2: Slope limiting
        try
        {
            std::vector<Set::Scalar> values = { 1.0, 2.0, 4.0, 3.0, 3.5 };

            // Monotonic region
            Set::Scalar slope1 = limiter.limitSlope(values, 2);
            if (std::abs(slope1 - 1.0) > 1E-10)
            {
                Util::Warning(INFO, "limitSlope at index 2 should be 1.0, got ", slope1);
                Util::Exception(INFO, "Slope limiting in monotonic region failed");
            }

            // Non-monotonic region (local maximum)
            Set::Scalar slope2 = limiter.limitSlope(values, 3);
            if (std::abs(slope2) > 1E-10)
            {
                Util::Warning(INFO, "limitSlope at index 3 should be 0.0, got ", slope2);
                Util::Exception(INFO, "Slope limiting at local extremum failed");
            }

            Util::Test::SubMessage("Test 2: Slope limiting", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 2: Slope limiting", 1);
        }

        // Test 3: State reconstruction
        try
        {
            std::vector<State> cells = {
                State(1.0, 1.0, 0.0, 1.0),
                State(2.0, 2.0, 0.0, 2.0),
                State(3.0, 3.0, 0.0, 3.0)
            };

            std::vector<State> leftStates, rightStates;
            limiter.reconstructStates(cells, leftStates, rightStates);

            // Check that reconstruction preserves cell averages
            for (size_t i = 1; i < cells.size(); i++)
            {
                Set::Scalar avg_rho = 0.5 * (leftStates[i].rho + rightStates[i].rho);
                if (std::abs(avg_rho - cells[i - 1].rho) > 1E-10)
                {
                    Util::Exception(INFO, "Reconstruction does not preserve cell average");
                }
            }

            Util::Test::SubMessage("Test 3: State reconstruction", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 3: State reconstruction", 1);
        }

        return failed;
    }
};

} // namespace Limiter
} // namespace Local
} // namespace Solver

#endif // SOLVER_LOCAL_LIMITER_MINMOD_H
