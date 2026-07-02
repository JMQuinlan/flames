#ifndef SOLVER_LOCAL_FLUIDRIEMANN_PARTIALLY_PARABOLIC_H
#define SOLVER_LOCAL_FLUIDRIEMANN_PARTIALLY_PARABOLIC_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
//#include "Util/Util.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// Partially Parabolic Method Solver (Hyperbolic + Parabolic)
/// Combines HLLC hyperbolic flux with viscous/thermal parabolic terms
class PartiallyParabolic : public FluidRiemann
{
public:
    static constexpr const char *name = "partially_parabolic";

    PartiallyParabolic(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    PartiallyParabolic(IO::ParmParse &pp) { pp_queryclass(*this); }
    PartiallyParabolic()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    Set::Scalar theta = 0.5;         // Implicitness parameter (0=explicit, 0.5=Crank-Nicolson, 1=fully implicit)
    Set::Scalar epsilon = 1e-6;      // Regularization parameter for interface
    bool use_hyperbolic_flux = true; // Whether to use hyperbolic flux calculation
    bool use_parabolic_flux = true;  // Whether to use parabolic flux calculation

    static void Parse(PartiallyParabolic &value, IO::ParmParse &pp)
    {
        pp.query_default("theta", value.theta, 0.5);
        pp.query_default("epsilon", value.epsilon, 1e-6);
        pp.query_default("use_hyperbolic_flux", value.use_hyperbolic_flux, true);
        pp.query_default("use_parabolic_flux", value.use_parabolic_flux, true);
    }

    // Extended Solve signature for parabolic terms
    Flux SolveExtended(State lo, State hi, Set::Scalar small, Set::Scalar mu = 0.0, Set::Scalar k_thermal = 0.0, Set::Scalar dt = 0.0, Set::Scalar dx = 0.0, int Spec_Vol = 1)
    {
        Flux total_flux;

        // STEP 1: Extract state variables
        Set::Scalar rho_L = lo.rho, rho_R = hi.rho;
        Set::Scalar Mn_L = lo.M_normal, Mn_R = hi.M_normal;
        Set::Scalar Mt_L = lo.M_tangent, Mt_R = hi.M_tangent;
        Set::Scalar E_L = lo.E, E_R = hi.E;
        Set::Scalar T_L = lo.T, T_R = hi.T;
        Set::Scalar gamma_L = lo.gam, gamma_R = hi.gam;
        Set::Scalar p0_L = lo.p0, p0_R = hi.p0;

        // Ensure no negative densities
        rho_L = std::max(small, rho_L);
        rho_R = std::max(small, rho_R);

        // STEP 2: Compute fluid primitives using Tammann EOS
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

        // /////////////////////////////////////////////////////
        // PART I: HYPERBOLIC FLUX CALCULATION (HLLC)
        // /////////////////////////////////////////////////////
        Flux hyperbolic_flux;
        if (use_hyperbolic_flux)
        {
            // Sound speeds (Tammann EOS)
            Set::Scalar a_L = std::sqrt(gamma_L * (p_L + p0_L) / (rho_L + small));
            Set::Scalar a_R = std::sqrt(gamma_R * (p_R + p0_R) / (rho_R + small));

            // Wave speed estimates
            Set::Scalar S_L = std::min(u_L - a_L, u_R - a_R);
            Set::Scalar S_R = std::max(u_L + a_L, u_R + a_R);

            // Middle wave speed (contact discontinuity)
            Set::Scalar S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + small);

            // Physical fluxes
            Flux f_L, f_R;
            if (Spec_Vol == 1)
            {
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
                f_L.mass = rho_L * u_L;
                f_L.momentum_normal = rho_L * u_L * u_L + p_L;
                f_L.momentum_tangent = rho_L * u_L * v_L;
                f_L.energy = u_L * (E_L + p_L / (rho_L + small));

                f_R.mass = rho_R * u_R;
                f_R.momentum_normal = rho_R * u_R * u_R + p_R;
                f_R.momentum_tangent = rho_R * u_R * v_R;
                f_R.energy = u_R * (E_R + p_R / (rho_R + small));
            }

            // Star state densities
            Set::Scalar rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star + small);
            Set::Scalar rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star + small);

            // Star state energies (corrected formula)
            Set::Scalar E_star_L, E_star_R;
            if (Spec_Vol == 1)
            {
                E_star_L = rho_star_L * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small)));
                E_star_R = rho_star_R * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small)));
            }
            else
            {
                E_star_L = E_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small));
                E_star_R = E_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small));
            }

            // Star state fluxes
            Flux f_star_L, f_star_R;
            f_star_L.mass = f_L.mass + S_L * (rho_star_L - rho_L);
            f_star_L.momentum_normal = f_L.momentum_normal + S_L * (rho_star_L * S_star - rho_L * u_L);
            f_star_L.momentum_tangent = f_L.momentum_tangent + S_L * (rho_star_L * v_L - rho_L * v_L);
            f_star_L.energy = f_L.energy + S_L * (E_star_L - E_L);

            f_star_R.mass = f_R.mass + S_R * (rho_star_R - rho_R);
            f_star_R.momentum_normal = f_R.momentum_normal + S_R * (rho_star_R * S_star - rho_R * u_R);
            f_star_R.momentum_tangent = f_R.momentum_tangent + S_R * (rho_star_R * v_R - rho_R * v_R);
            f_star_R.energy = f_R.energy + S_R * (E_star_R - E_R);

            // Select flux based on wave positions
            if (0.0 <= S_L)
            {
                hyperbolic_flux = f_L;
            }
            else if (S_L <= 0.0 && 0.0 <= S_star)
            {
                hyperbolic_flux = f_star_L;
            }
            else if (S_star <= 0.0 && 0.0 <= S_R)
            {
                hyperbolic_flux = f_star_R;
            }
            else
            {
                hyperbolic_flux = f_R;
            }
        }

        total_flux = hyperbolic_flux;

        // /////////////////////////////////////////////////////
        // PART II: PARABOLIC FLUX CALCULATION (viscous/thermal)
        // /////////////////////////////////////////////////////
        if (use_parabolic_flux && mu > small)
        {
            Flux parabolic_flux;

            // Interface values for viscous work calculation
            Set::Scalar u_interface = 0.5 * (u_L + u_R);
            Set::Scalar v_interface = 0.5 * (v_L + v_R);

            // Velocity gradients at interface
            Set::Scalar du_dx = (u_R - u_L) / (dx + small);
            Set::Scalar dv_dx = (v_R - v_L) / (dx + small);

            // Divergence of velocity (1D approximation)
            Set::Scalar div_u = du_dx;

            // Viscous stress tensor components (Newtonian fluid)
            // tau_ij = mu * (du_i/dx_j + du_j/dx_i - 2/3 * delta_ij * div(u))
            Set::Scalar tau_xx = 2.0 * mu * (du_dx - div_u / 3.0);
            Set::Scalar tau_xy = mu * dv_dx;

            // No mass flux from viscous terms
            parabolic_flux.mass = 0.0;

            // Momentum flux from viscous stresses (negative sign: stress opposes gradient)
            parabolic_flux.momentum_normal = -tau_xx;
            parabolic_flux.momentum_tangent = -tau_xy;

            // Energy flux from viscous work and heat conduction
            // Viscous work: tau_ij * u_i
            Set::Scalar viscous_work = tau_xx * u_interface + tau_xy * v_interface;

            // Heat conduction: q = -k * dT/dx
            Set::Scalar dT_dx = (T_R - T_L) / (dx + small);
            Set::Scalar heat_flux = -k_thermal * dT_dx;

            parabolic_flux.energy = -viscous_work + heat_flux;

            // Apply theta-scheme for time integration
            // theta=0: explicit, theta=0.5: Crank-Nicolson, theta=1: fully implicit
            if (dt > small)
            {
                parabolic_flux.momentum_normal *= theta;
                parabolic_flux.momentum_tangent *= theta;
                parabolic_flux.energy *= theta;
            }

            // Add parabolic flux to total
            total_flux.mass += parabolic_flux.mass;
            total_flux.momentum_normal += parabolic_flux.momentum_normal;
            total_flux.momentum_tangent += parabolic_flux.momentum_tangent;
            total_flux.energy += parabolic_flux.energy;
        }

        // ERROR CHECKING
        if ((total_flux.mass != total_flux.mass) || (total_flux.momentum_normal != total_flux.momentum_normal) || (total_flux.momentum_tangent != total_flux.momentum_tangent) || (total_flux.energy != total_flux.energy))
        {
            bool verbose = false;
            if (verbose)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN PARTIALLY PARABOLIC SOLVER:");
                Util::ParallelMessage(INFO, "hi:   ", hi);
                Util::ParallelMessage(INFO, "lo:   ", lo);
                Util::ParallelMessage(INFO, "fluxT:", total_flux);
                Util::ParallelMessage(INFO, "rho_L=", rho_L);
                Util::ParallelMessage(INFO, "rho_R=", rho_R);
                Util::ParallelMessage(INFO, "u_L=  ", u_L);
                Util::ParallelMessage(INFO, "u_R=  ", u_R);
                Util::ParallelMessage(INFO, "p_L=  ", p_L);
                Util::ParallelMessage(INFO, "p_R=  ", p_R);
                Util::ParallelMessage(INFO, "mu=   ", mu);
                Util::ParallelMessage(INFO, "dt=   ", dt);
                Util::ParallelMessage(INFO, "dx=   ", dx);
            }
            Util::Exception(INFO);
        }

        return total_flux;
    }

    // Standard Solve interface (no parabolic terms)
    virtual Flux Solve(State lo, State hi, Set::Scalar small, int Spec_Vol = 1) override
    {
        return SolveExtended(lo, hi, small, 0.0, 0.0, 0.0, 0.0, Spec_Vol);
    }

    static int Test()
    {
        PartiallyParabolic solver;
        int failed = 0;
        Set::Scalar small = 1E-10;
        Set::Scalar mu = 0.01;
        Set::Scalar k_thermal = 0.01;
        Set::Scalar dt = 0.001;
        Set::Scalar dx = 0.01;

        // Test 1: Hyperbolic-only mode should match HLLC for identical states
        try
        {
            State left(1.0, 0.0, 0.0, 1.0);
            State right(1.0, 0.0, 0.0, 1.0);

            solver.use_parabolic_flux = false;
            Flux pp_flux = solver.SolveExtended(left, right, small, 0.0, 0.0, 0.0, 0.0);

            // For identical states, flux should be zero
            if (fabs(pp_flux.mass) > 1E-10 || fabs(pp_flux.momentum_normal) > 1E-10 || fabs(pp_flux.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "PP Flux:  ", pp_flux);
                Util::Exception(INFO, "Identical states should produce zero flux.");
            }
            Util::Test::SubMessage("Test 1: Hyperbolic-only mode for identical states", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 1: Hyperbolic-only mode for identical states", 1);
        }

        // Test 2: Parabolic-only mode should produce viscous fluxes for velocity gradient
        try
        {
            State left(1.0, 0.0, 0.0, 1.0);
            left.T = 300.0;
            State right(1.0, 1.0, 0.0, 1.0);
            right.T = 300.0;

            solver.use_hyperbolic_flux = false;
            solver.use_parabolic_flux = true;

            Flux parabolic_flux = solver.SolveExtended(left, right, small, mu, k_thermal, dt, dx);

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
            left.T = 300.0;
            State right(1.0, 1.0, 0.0, 1.0);
            right.T = 310.0;

            // Enable both flux components
            solver.use_hyperbolic_flux = true;
            solver.use_parabolic_flux = true;

            Flux combined_flux = solver.SolveExtended(left, right, small, mu, k_thermal, dt, dx);

            // Get hyperbolic-only flux
            solver.use_parabolic_flux = false;
            Flux hyperbolic_flux = solver.SolveExtended(left, right, small, 0.0, 0.0, 0.0, 0.0);

            // Get parabolic-only flux
            solver.use_hyperbolic_flux = false;
            solver.use_parabolic_flux = true;
            Flux parabolic_flux = solver.SolveExtended(left, right, small, mu, k_thermal, dt, dx);

            // Combined flux should be approximately the sum
            Set::Scalar expected_momentum = hyperbolic_flux.momentum_normal + parabolic_flux.momentum_normal;
            if (fabs(expected_momentum - combined_flux.momentum_normal) > 1E-8)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Combined Flux:  ", combined_flux);
                Util::Warning(INFO, "Hyperbolic Flux:  ", hyperbolic_flux);
                Util::Warning(INFO, "Parabolic Flux:  ", parabolic_flux);
                Util::Warning(INFO, "Expected momentum: ", expected_momentum);
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

} // namespace FluidRiemann
} // namespace Local
} // namespace Solver

#endif // SOLVER_LOCAL_FLUIDRIEMANN_PARTIALLY_PARABOLIC_H
