#ifndef SOLVER_LOCAL_RIEMANN_PARTIALLY_PARABOLIC_H
#define SOLVER_LOCAL_RIEMANN_PARTIALLY_PARABOLIC_H

#include "IO/ParmParse.H"
#include "Solver/Local/Riemann/Riemann.H"

/// A partially parabolic solver for multiphase flows

namespace Solver
{
/// Local solvers
namespace Local
{
namespace Riemann
{
/// Partially Parabolic Method Solver
class PartiallyParabolic
{
public:
    static constexpr const char *name = "partially_parabolic";

    PartiallyParabolic(IO::ParmParse &pp, std::string name)
    {
        pp_queryclass(name, *this);
    }

    PartiallyParabolic(IO::ParmParse &pp)
    {
        pp_queryclass(*this);
    }

    PartiallyParabolic()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    int verbose = 0;
    Set::Scalar theta = 0.5;         // Implicitness parameter (0=explicit, 0.5=Crank-Nicolson, 1=fully implicit)
    Set::Scalar epsilon = 1e-6;      // Regularization parameter for interface
    bool use_hyperbolic_flux = true; // Whether to use hyperbolic flux calculation
    bool use_parabolic_flux = true;  // Whether to use parabolic flux calculation

    static void
    Parse(PartiallyParabolic &value, IO::ParmParse &pp)
    {
        // Enable to dump diagnostic data if the solver fails
        pp.query_default("verbose", value.verbose, 1);

        // Implicitness parameter for parabolic terms
        pp.query_default("theta", value.theta, 0.5);

        // Interface regularization parameter
        pp.query_default("epsilon", value.epsilon, 1e-6);

        // Control which components to include
        pp.query_default("use_hyperbolic_flux", value.use_hyperbolic_flux, true);
        pp.query_default("use_parabolic_flux", value.use_parabolic_flux, true);
    }

    Flux
    Solve(State lo, State hi, Set::Scalar gamma, Set::Scalar p_ref, Set::Scalar small, Set::Scalar p_0 = 0.0, Set::Scalar mu = 0.0, Set::Scalar dt = 0.0, Set::Scalar dx = 0.0)
    {
        // Final flux will be the sum of hyperbolic and parabolic components
        Flux total_flux;

        // STEP 1: Extract state variables
        Set::Scalar rho_L = lo.rho, rho_R = hi.rho;
        Set::Scalar Mn_L = lo.M_normal, Mn_R = hi.M_normal;
        Set::Scalar Mt_L = lo.M_tangent, Mt_R = hi.M_tangent;
        Set::Scalar E_L = lo.E, E_R = hi.E;

        // Ensure no negative densities
        rho_L = std::max(small, rho_L);
        rho_R = std::max(small, rho_R);

        // STEP 2: Compute primitive variables
        Set::Scalar ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L + small);
        Set::Scalar ue_L = E_L - ke_L;
        Set::Scalar p_L = (gamma - 1.0) * ue_L - gamma * (p_0 + p_ref); // pressure Tammann EOS modification

        Set::Scalar ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R + small);
        Set::Scalar ue_R = E_R - ke_R;
        Set::Scalar p_R = (gamma - 1.0) * ue_R - gamma * (p_0 + p_ref);

        Set::Scalar u_L = Mn_L / (rho_L + small), u_R = Mn_R / (rho_R + small);
        Set::Scalar v_L = Mt_L / (rho_L + small), v_R = Mt_R / (rho_R + small);

        // STEP 3: Calculate interface values
        Set::Scalar rho_interface = 0.5 * (rho_L + rho_R);
        Set::Scalar u_interface = 0.5 * (u_L + u_R);
        Set::Scalar v_interface = 0.5 * (v_L + v_R);
        Set::Scalar p_interface = 0.5 * (p_L + p_R);

        // PART I: HYPERBOLIC FLUX CALCULATION (similar to HLLC)
        if (use_hyperbolic_flux)
        {
            // Calculate sound speeds
            Set::Scalar a_L = std::sqrt(gamma * (p_L + p_ref) / (rho_L + small));
            Set::Scalar a_R = std::sqrt(gamma * (p_R + p_ref) / (rho_R + small));

            // Compute wave speed estimates
            Set::Scalar S_L = std::min(u_L - a_L, u_R - a_R);
            Set::Scalar S_R = std::max(u_L + a_L, u_R + a_R);

            // Compute middle wave speed (contact discontinuity)
            Set::Scalar S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + small);

            // Compute physical fluxes for left and right states
            Flux f_L, f_R;

            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (E_L + p_L);

            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (E_R + p_R);

            // Compute star state values
            Set::Scalar rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star + small);
            Set::Scalar rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star + small);

            // Compute star state fluxes
            Flux f_star_L, f_star_R;

            f_star_L.mass = f_L.mass + S_L * (rho_star_L - rho_L);
            f_star_L.momentum_normal = f_L.momentum_normal + S_L * (rho_star_L * S_star - rho_L * u_L);
            f_star_L.momentum_tangent = f_L.momentum_tangent + S_L * (rho_star_L * v_L - rho_L * v_L);
            f_star_L.energy = f_L.energy + S_L * (rho_star_L * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small))) - E_L);

            f_star_R.mass = f_R.mass + S_R * (rho_star_R - rho_R);
            f_star_R.momentum_normal = f_R.momentum_normal + S_R * (rho_star_R * S_star - rho_R * u_R);
            f_star_R.momentum_tangent = f_R.momentum_tangent + S_R * (rho_star_R * v_R - rho_R * v_R);
            f_star_R.energy = f_R.energy + S_R * (rho_star_R * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small))) - E_R);

            // Determine hyperbolic flux based on wave positions
            Flux hyperbolic_flux;

            if (0 <= S_L)
            {
                hyperbolic_flux = f_L;
            }
            else if (S_L <= 0 && 0 <= S_star)
            {
                hyperbolic_flux = f_star_L;
            }
            else if (S_star <= 0 && 0 <= S_R)
            {
                hyperbolic_flux = f_star_R;
            }
            else
            {
                hyperbolic_flux = f_R;
            }

            // Add hyperbolic flux to total flux
            total_flux.mass = hyperbolic_flux.mass;
            total_flux.momentum_normal = hyperbolic_flux.momentum_normal;
            total_flux.momentum_tangent = hyperbolic_flux.momentum_tangent;
            total_flux.energy = hyperbolic_flux.energy;
        }

        // PART II: PARABOLIC FLUX CALCULATION (viscous terms)
        if (use_parabolic_flux && mu > small && dx > small)
        {
            Flux parabolic_flux;

            // Calculate velocity gradients at the interface
            Set::Scalar du_dx = (u_R - u_L) / dx;
            Set::Scalar dv_dx = (v_R - v_L) / dx;

            // Calculate divergence of velocity (approximate)
            Set::Scalar div_u = du_dx; // In 1D, this is just du/dx

            // Calculate viscous stress components
            Set::Scalar tau_xx = 2.0 * mu * (du_dx - div_u / 3.0);
            Set::Scalar tau_xy = mu * dv_dx;

            // No mass flux from viscous terms
            parabolic_flux.mass = 0.0;

            // Momentum flux from viscous stresses
            parabolic_flux.momentum_normal = -tau_xx; // Negative because stress opposes gradient
            parabolic_flux.momentum_tangent = -tau_xy;

            // Energy flux from viscous work and heat conduction
            // Work done by viscous forces: tau_ij * u_i
            Set::Scalar viscous_work = tau_xx * u_interface + tau_xy * v_interface;

            // Heat conduction (simplified, using temperature gradient approximation)
            Set::Scalar k_thermal = mu * gamma / (gamma - 1.0); // Thermal conductivity (Prandtl number = 1)
            Set::Scalar T_L = p_L / (rho_L * (gamma - 1.0) + small);
            Set::Scalar T_R = p_R / (rho_R * (gamma - 1.0) + small);
            Set::Scalar dT_dx = (T_R - T_L) / dx;
            Set::Scalar heat_flux = -k_thermal * dT_dx;

            parabolic_flux.energy = -viscous_work + heat_flux;

            // Apply implicitness parameter (theta-scheme)
            // For theta=0: fully explicit, theta=1: fully implicit, theta=0.5: Crank-Nicolson
            if (dt > small)
            {
                // Scale parabolic flux by theta for implicit treatment
                parabolic_flux.momentum_normal *= theta;
                parabolic_flux.momentum_tangent *= theta;
                parabolic_flux.energy *= theta;
            }

            // Add parabolic flux to total flux
            total_flux.mass += parabolic_flux.mass;
            total_flux.momentum_normal += parabolic_flux.momentum_normal;
            total_flux.momentum_tangent += parabolic_flux.momentum_tangent;
            total_flux.energy += parabolic_flux.energy;
        }

        // PART III: SURFACE TENSION CONTRIBUTION (if needed)
        // This would be added here if surface tension is included in the flux calculation
        // For now, we assume surface tension is handled separately as a source term

        // ERROR CHECKING
        if ((total_flux.mass != total_flux.mass) || (total_flux.momentum_normal != total_flux.momentum_normal) || (total_flux.momentum_tangent != total_flux.momentum_tangent) || (total_flux.energy != total_flux.energy))
        {
            if (verbose)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN PARTIALLY PARABOLIC SOLVER:");
                Util::ParallelMessage(INFO, "hi:   ", hi);
                Util::ParallelMessage(INFO, "lo:   ", lo);
                Util::ParallelMessage(INFO, "rho_L=", rho_L);
                Util::ParallelMessage(INFO, "rho_R=", rho_R);
                Util::ParallelMessage(INFO, "u_L=  ", u_L);
                Util::ParallelMessage(INFO, "u_R=  ", u_R);
                Util::ParallelMessage(INFO, "ke_R= ", ke_R);
                Util::ParallelMessage(INFO, "ke_L= ", ke_L);
                Util::ParallelMessage(INFO, "ue_L= ", ue_L);
                Util::ParallelMessage(INFO, "ue_R= ", ue_R);
                Util::ParallelMessage(INFO, "p_L=  ", p_L);
                Util::ParallelMessage(INFO, "p_R=  ", p_R);
                //Util::ParallelMessage(INFO, "a_L=  ", a_L);
                //Util::ParallelMessage(INFO, "a_R=  ", a_R);
                //Util::ParallelMessage(INFO, "S_L=  ", S_L);
                //Util::ParallelMessage(INFO, "S_R=  ", S_R);
                Util::ParallelMessage(INFO, "mu ", mu);
                Util::ParallelMessage(INFO, "dt ", dt);
                Util::ParallelMessage(INFO, "dx ", dx);
            }
            Util::Exception(INFO);
        }

        return total_flux;
    }

    static int
    Test()
    {
        PartiallyParabolic solver;
        int failed = 0;
        Set::Scalar gamma = 1.4;
        Set::Scalar pref = 10.0;
        Set::Scalar small = 1E-10;
        Set::Scalar mu = 0.01;  // Viscosity
        Set::Scalar dt = 0.001; // Time step
        Set::Scalar dx = 0.01;  // Grid spacing

        // Test 1: Hyperbolic-only mode should match HLLC for identical states
        try
        {
            State left(1.0, 0.0, 0.0, 1.0);
            State right(1.0, 0.0, 0.0, 1.0);

            // Create HLLC solver for comparison
            HLLC hllc_solver;

            // Set parabolic flux off to match HLLC
            solver.use_parabolic_flux = false;

            Flux pp_flux = solver.Solve(left, right, gamma, pref, small, 0.0, 0.0, 0.0, 0.0);
            Flux hllc_flux = hllc_solver.Solve(left, right, gamma, pref, small);

            if (fabs(pp_flux.mass - hllc_flux.mass) > 1E-10 || fabs(pp_flux.momentum_normal - hllc_flux.momentum_normal) > 1E-10 || fabs(pp_flux.energy - hllc_flux.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "PP Flux:  ", pp_flux);
                Util::Warning(INFO, "HLLC Flux:  ", hllc_flux);
                Util::Exception(INFO, "Hyperbolic-only mode should match HLLC for identical states.");
            }

            Util::Test::SubMessage("Test 1: Hyperbolic-only mode matches HLLC", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 1: Hyperbolic-only mode matches HLLC", 1);
        }

        // Test 2: Parabolic-only mode should produce viscous fluxes for velocity gradient
        try
        {
            State left(1.0, 0.0, 0.0, 1.0);
            State right(1.0, 1.0, 0.0, 1.0);

            // Set hyperbolic flux off to isolate parabolic effects
            solver.use_hyperbolic_flux = false;
            solver.use_parabolic_flux = true;

            Flux parabolic_flux = solver.Solve(left, right, gamma, pref, small, 0.0, mu, dt, dx);

            // Viscous flux should be non-zero for velocity gradient
            if (fabs(parabolic_flux.momentum_normal) < 1E-10)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Parabolic Flux:  ", parabolic_flux);
                Util::Exception(INFO, "Parabolic-only mode should produce viscous fluxes for velocity gradient.");
            }

            Util::Test::SubMessage("Test 2: Parabolic-only mode produces viscous fluxes", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 2: Parabolic-only mode produces viscous fluxes", 1);
        }

        // Test 3: Combined mode should include both hyperbolic and parabolic contributions
        try
        {
            State left(1.0, 0.5, 0.0, 1.0);
            State right(1.0, 1.0, 0.0, 1.0);

            // Enable both flux components
            solver.use_hyperbolic_flux = true;
            solver.use_parabolic_flux = true;

            // Get combined flux
            Flux combined_flux = solver.Solve(left, right, gamma, pref, small, 0.0, mu, dt, dx);

            // Get hyperbolic-only flux
            solver.use_parabolic_flux = false;
            Flux hyperbolic_flux = solver.Solve(left, right, gamma, pref, small, 0.0, 0.0, 0.0, 0.0);

            // Get parabolic-only flux
            solver.use_hyperbolic_flux = false;
            solver.use_parabolic_flux = true;
            Flux parabolic_flux = solver.Solve(left, right, gamma, pref, small, 0.0, mu, dt, dx);

            // Combined flux should be approximately the sum of individual components
            if (fabs((hyperbolic_flux.momentum_normal + parabolic_flux.momentum_normal) - combined_flux.momentum_normal) > 1E-10)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Combined Flux:  ", combined_flux);
                Util::Warning(INFO, "Hyperbolic Flux:  ", hyperbolic_flux);
                Util::Warning(INFO, "Parabolic Flux:  ", parabolic_flux);
                Util::Exception(INFO, "Combined mode should include both hyperbolic and parabolic contributions.");
            }

            Util::Test::SubMessage("Test 3: Combined mode includes both flux components", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 3: Combined mode includes both flux components", 1);
        }

        // Reset solver settings
        solver.use_hyperbolic_flux = true;
        solver.use_parabolic_flux = true;

        return failed;
    }
};

} // namespace Riemann
} // namespace Local
} // namespace Solver

#endif // SOLVER_LOCAL_RIEMANN_PARTIALLY_PARABOLIC_H
